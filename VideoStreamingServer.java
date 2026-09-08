import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

public class VideoStreamingServer {

    private static final int DEFAULT_PORT  = 9090;
    private static final int TARGET_HEIGHT = 360;
    private static final int TARGET_WIDTH  = 640;
    /**
     * Shared panorama height. Panels are rotated/shifted onto this canvas so a
     * common horizon can sit on one row; empty corners stay black.
     */
    private static final int ALIGNED_HEIGHT = 540;
    /** Horizontal overlap between adjacent panels (wider = more blended seams). */
    private static final int OVERLAP_PX    = 168;
    /** Horizontal FOV of each rectified panel. Higher = more zoomed out. */
    private static final double OUTPUT_FOV_DEG = 128.0;
    /** Keep this fraction of the remapped frame (1.0 = no extra zoom crop). */
    private static final double CROP_WIDTH_FRACTION = 0.96;
    private static final double CROP_MAX_HEIGHT_FRACTION = 0.94;
    /** 0 = keep the top of the remap, 1 = keep the bottom (ground). */
    private static final double CROP_Y_BIAS = 0.50;
    /** Drop rows/cols darker than this after remap (fisheye rim / empty map). */
    private static final double VALID_LUMA_MIN = 14.0;
    /** Extra inset of the valid region so the curved fisheye rim is not stretched. */
    private static final double VALID_INSET_FRACTION = 0.015;
    /** Horizon row in the shared output frame (fraction of height from the top). */
    private static final double HORIZON_FRACTION = 0.38;
    /** Last stitch panel — rear camera (bumper at bottom of raw fisheye). */
    private static final int REAR_CAMERA_INDEX = 3;
    /** Max corrective roll when a camera is physically tilted (rear often is). */
    private static final double MAX_ROLL_DEG = 40.0;
    /** Max vertical shift as a fraction of the rectified panel height. */
    private static final double MAX_HORIZON_SHIFT_FRACTION = 0.42;

    // =========================================================================
    public static void main(String[] args) throws IOException {

        Path leftVideo  = ensureDecodable(Paths.get("left_1.mp4"));
        Path frontVideo = ensureDecodable(Paths.get("front_1.mp4"));
        Path rightVideo = ensureDecodable(Paths.get("right_1.mp4"));
        Path backVideo  = ensureDecodable(Paths.get("rear_1.mp4"));

        if (args.length >= 5) {
            leftVideo  = ensureDecodable(Paths.get(args[1]));
            frontVideo = ensureDecodable(Paths.get(args[2]));
            rightVideo = ensureDecodable(Paths.get(args[3]));
            backVideo  = ensureDecodable(Paths.get(args[4]));
        }

        Path[] videos = { leftVideo, frontVideo, rightVideo, backVideo };

        for (Path v : videos) {
            if (!Files.exists(v) || Files.isDirectory(v)) {
                System.err.println("Video file not found: " + v);
                return;
            }
        }

        int port = args.length > 0 ? Integer.parseInt(args[0]) : DEFAULT_PORT;

        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("OpenCV native library not found: " + e.getMessage());
            return;
        }
        loadFfmpegPlugin();

        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);

        server.createContext("/stitch", new StitchHandler(videos));
        server.createContext("/play",   new PlayerPageHandler());

        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();

        System.out.println("Server started  ->  http://localhost:" + port + "/play");
        System.out.println("Feeds: left=" + leftVideo + " front=" + frontVideo
                + " right=" + rightVideo + " rear=" + backVideo);
    }

    // STITCH HANDLER  -  undistort each feed, then feather-blend panorama

    private static class StitchHandler implements HttpHandler {
      private final Path[] videoFiles;
      private double[] seamDy;
      StitchHandler(Path[] f) { this.videoFiles = f; }

      @Override public void handle(HttpExchange ex) throws IOException {
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) {
          ex.sendResponseHeaders(405, -1); return;
        }

        VideoCapture[] caps = new VideoCapture[videoFiles.length];
        for (int i = 0; i < videoFiles.length; i++) {
          caps[i] = openVideo(videoFiles[i]);
          if (caps[i] == null || !caps[i].isOpened()) {
            System.err.println("Could not open video: " + videoFiles[i]);
            ex.sendResponseHeaders(500, -1); return;
          }
        }

        CameraFeedFilter[] undistort = new CameraFeedFilter[videoFiles.length];
        for (int i = 0; i < undistort.length; i++) {
          undistort[i] = FisheyePanelFilter.forStitchIndex(i);
        }

        ex.getResponseHeaders().set("Content-Type", "multipart/x-mixed-replace; boundary=frame");
        ex.sendResponseHeaders(200, 0);

        try (OutputStream out = ex.getResponseBody()) {
          Mat[] frames  = new Mat[videoFiles.length];
          Mat[] ready   = new Mat[videoFiles.length];
          for (int i = 0; i < videoFiles.length; i++) {
            frames[i] = new Mat();
            ready[i]  = new Mat();
          }

          while (true) {
            for (int i = 0; i < caps.length; i++) {
              if (!readOrLoop(caps, i, videoFiles[i], frames[i])) {
                continue;
              }
              undistort[i].recalibrateAndFilter(frames[i], ready[i]);
            }

            boolean allReady = true;
            for (int i = 0; i < ready.length; i++) {
              if (ready[i].empty()
                  || ready[i].cols() != TARGET_WIDTH
                  || ready[i].rows() != ALIGNED_HEIGHT) {
                allReady = false;
                break;
              }
            }
            if (!allReady) {
              continue;
            }

            if (seamDy == null) {
              seamDy = estimatePairwiseDy(ready);
            }
            Mat[] aligned = applyVerticalShifts(ready, seamDy);
            Mat panorama = featherStitch(aligned);
            if (aligned != ready) {
              for (Mat m : aligned) {
                m.release();
              }
            }
            writeFrame(out, encodeJpeg(panorama));
            panorama.release();
          }
        }
        finally {
          for (VideoCapture c : caps) c.release();
        }

      } // handle()
    } // StitchHandler


    // PLAYER PAGE  -  stitched panorama only, full-viewport

    private static class PlayerPageHandler implements HttpHandler {

        @Override public void handle(HttpExchange ex) throws IOException {
            String html = "<!DOCTYPE html><html lang='en'><head>"
                + "<meta charset='UTF-8'>"
                + "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                + "<title>360° Panoramic View</title>"
                + "<style>"
                + "*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }"
                + "html, body { height: 100%; background: #0a0a0f; color: #e0e0e0;"
                + "  font-family: 'Segoe UI', sans-serif; overflow: hidden; }"
                + ".container { display: flex; flex-direction: column;"
                + "  align-items: center; justify-content: center;"
                + "  height: 100vh; padding: 16px; gap: 12px; }"
                + "h1 { font-size: 1.4rem; font-weight: 300; letter-spacing: 2px;"
                + "  color: #7ec8e3; text-align: center; flex-shrink: 0; }"
                + ".pano-wrap { width: 100%; flex: 1; min-height: 0;"
                + "  border: 1px solid #2a2a3a; border-radius: 8px; overflow: hidden;"
                + "  display: flex; align-items: center; justify-content: center; }"
                + ".pano-wrap img { width: 100%; height: 100%; object-fit: contain; display: block; }"
                + "</style>"
                + "</head><body>"
                + "<div class='container'>"
                + "  <h1>360° Panoramic Camera System</h1>"
                + "  <div class='pano-wrap'>"
                + "    <img src='/stitch' alt='360° stitched panorama'>"
                + "  </div>"
                + "</div>"
                + "</body></html>";

            byte[] bytes = html.getBytes("UTF-8");
            ex.getResponseHeaders().set("Content-Type", "text/html; charset=UTF-8");
            ex.sendResponseHeaders(200, bytes.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
        }
    }


    interface CameraFeedFilter {
        void recalibrateAndFilter(Mat src, Mat dst360x640);
    }

    /**
     * Mount pose for one surround camera. Pitch is virtual-camera tilt in the
     * OpenCV Y-down frame: negative looks toward the top of the raw fisheye
     * (street instead of bumper when the lens points at the ground).
     */
    static final class PanelPose {
        final double pitchDeg;
        final double yawDeg;
        final double rollDeg;
        final double inputFovDeg;
        final double outputFovDeg;

        PanelPose(double pitchDeg, double yawDeg, double rollDeg,
                  double inputFovDeg, double outputFovDeg) {
            this.pitchDeg = pitchDeg;
            this.yawDeg = yawDeg;
            this.rollDeg = rollDeg;
            this.inputFovDeg = inputFovDeg;
            this.outputFovDeg = outputFovDeg;
        }
    }

    //  UNIFIED FISHEYE PANEL
    //
    //  Every feed is remapped with the same output size and FOV, then cropped
    //  to the filled rectangle (drops circular vignette). Each panel is then
    //  rotated and shifted onto a shared canvas so one horizon line is common;
    //  empty regions stay black. Adjacent panels are feather-blended.

    static final class FisheyePanelFilter implements CameraFeedFilter {

        private static final double[] FISHEYE_D = { 0.0, 0.0, 0.0, 0.0 };
        private static final int WORK_HEIGHT = TARGET_HEIGHT * 2;
        private static final int WORK_WIDTH  = TARGET_WIDTH * 2;

        private final PanelPose pose;
        private Mat map1;
        private Mat map2;
        private Mat undistorted;
        private Mat cropped;
        private Mat aligned;
        private Mat affine;
        private Rect workCrop;
        private int cachedSrcW = -1;
        private int cachedSrcH = -1;

        static FisheyePanelFilter forStitchIndex(int index) {
            final double inFov  = 160.0;
            if (index == REAR_CAMERA_INDEX) {
                return new FisheyePanelFilter(new PanelPose(-14.0, 0.0, 0.0, inFov, OUTPUT_FOV_DEG));
            }
            return new FisheyePanelFilter(new PanelPose(-6.0, 0.0, 0.0, inFov, OUTPUT_FOV_DEG));
        }

        FisheyePanelFilter(PanelPose pose) {
            this.pose = pose;
        }

        @Override
        public void recalibrateAndFilter(Mat src, Mat dst360x640) {
            if (src == null || src.empty()) {
                return;
            }

            ensureMaps(src.cols(), src.rows());

            if (undistorted == null) undistorted = new Mat();
            if (cropped == null)     cropped     = new Mat();
            if (aligned == null)     aligned     = new Mat();

            Imgproc.remap(src, undistorted, map1, map2, Imgproc.INTER_LINEAR,
                    Core.BORDER_CONSTANT, new Scalar(0, 0, 0));

            if (workCrop == null) {
                workCrop = cropWindow(undistorted);
            }
            Mat workRoi = undistorted.submat(workCrop);
            Imgproc.resize(workRoi, cropped, new Size(TARGET_WIDTH, TARGET_HEIGHT),
                    0, 0, Imgproc.INTER_AREA);
            workRoi.release();

            if (affine == null) {
                learnGroundLock(cropped);
            }

            // Rotate/translate onto the shared canvas. Do not recrop: empty
            // corners from the correction stay black so the horizon stays locked.
            Imgproc.warpAffine(cropped, aligned, affine,
                    new Size(TARGET_WIDTH, ALIGNED_HEIGHT),
                    Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
            aligned.copyTo(dst360x640);
            forceExactSize(dst360x640, TARGET_WIDTH, ALIGNED_HEIGHT);
        }

        private void learnGroundLock(Mat panel) {
            double roll = estimateRollDeg(panel);
            if (Math.abs(roll) > MAX_ROLL_DEG) {
                roll = Math.copySign(MAX_ROLL_DEG, roll);
            }

            Point center = new Point(panel.cols() / 2.0, panel.rows() / 2.0);
            Mat rot = Imgproc.getRotationMatrix2D(center, roll, 1.0);
            Mat leveled = new Mat();
            Imgproc.warpAffine(panel, leveled, rot,
                    new Size(panel.cols(), panel.rows()),
                    Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));

            int horizon = estimateHorizonRow(leveled);
            int targetY = (int) Math.round(HORIZON_FRACTION * ALIGNED_HEIGHT);
            double dy = targetY - horizon;
            double maxShift = TARGET_HEIGHT * MAX_HORIZON_SHIFT_FRACTION;
            if (dy > maxShift) dy = maxShift;
            if (dy < -maxShift) dy = -maxShift;

            affine = rot;
            affine.put(0, 2, affine.get(0, 2)[0]
                    + (TARGET_WIDTH - panel.cols()) / 2.0);
            affine.put(1, 2, affine.get(1, 2)[0] + dy);
            leveled.release();
        }

        private void ensureMaps(int srcW, int srcH) {
            if (map1 != null && srcW == cachedSrcW && srcH == cachedSrcH) {
                return;
            }

            Size dstSize = new Size(WORK_WIDTH, WORK_HEIGHT);

            Mat K = equidistantK(srcW, srcH, pose.inputFovDeg);
            Mat D = distortionCoeffs();
            Mat R = eulerRyxz(pose.pitchDeg, pose.yawDeg, pose.rollDeg);
            Mat P = pinholeK(WORK_WIDTH, WORK_HEIGHT, pose.outputFovDeg);
            P.put(1, 2, HORIZON_FRACTION * WORK_HEIGHT);

            if (map1 == null) map1 = new Mat();
            if (map2 == null) map2 = new Mat();

            Calib3d.fisheye_initUndistortRectifyMap(
                    K, D, R, P, dstSize, CvType.CV_16SC2, map1, map2);

            cachedSrcW = srcW;
            cachedSrcH = srcH;
            affine = null;
            workCrop = null;

            K.release();
            D.release();
            R.release();
            P.release();
        }

        static Mat equidistantK(int width, int height, double fovDeg) {
            double half = Math.toRadians(fovDeg) / 2.0;
            double f = (Math.min(width, height) / 2.0) / half;
            return matrixK(f, f, width / 2.0, height / 2.0);
        }

        static Mat pinholeK(int width, int height, double fovDeg) {
            double half = Math.toRadians(fovDeg) / 2.0;
            double f = (width / 2.0) / Math.tan(half);
            return matrixK(f, f, width / 2.0, height / 2.0);
        }

        static Mat matrixK(double fx, double fy, double cx, double cy) {
            Mat K = Mat.eye(3, 3, CvType.CV_64FC1);
            K.put(0, 0, fx);
            K.put(1, 1, fy);
            K.put(0, 2, cx);
            K.put(1, 2, cy);
            return K;
        }

        static Mat distortionCoeffs() {
            Mat D = new Mat(4, 1, CvType.CV_64FC1);
            D.put(0, 0, FISHEYE_D[0]);
            D.put(1, 0, FISHEYE_D[1]);
            D.put(2, 0, FISHEYE_D[2]);
            D.put(3, 0, FISHEYE_D[3]);
            return D;
        }

        static Mat eulerRyxz(double pitchDeg, double yawDeg, double rollDeg) {
            Mat rvec = new Mat(3, 1, CvType.CV_64FC1);
            rvec.put(0, 0, Math.toRadians(pitchDeg));
            rvec.put(1, 0, Math.toRadians(yawDeg));
            rvec.put(2, 0, Math.toRadians(rollDeg));
            Mat R = new Mat();
            Calib3d.Rodrigues(rvec, R);
            rvec.release();
            return R;
        }
    }


    static Rect cropWindow(Mat src) {
        Rect valid = validPixelRect(src);
        double aspect = (double) TARGET_WIDTH / TARGET_HEIGHT;
        int imgW = src.cols();
        int imgH = src.rows();

        int cropW = Math.max(2, (int) Math.round(imgW * CROP_WIDTH_FRACTION));
        int cropH = Math.max(2, (int) Math.round(cropW / aspect));
        if (cropH > imgH * CROP_MAX_HEIGHT_FRACTION) {
            cropH = Math.max(2, (int) Math.round(imgH * CROP_MAX_HEIGHT_FRACTION));
            cropW = Math.max(2, (int) Math.round(cropH * aspect));
        }

        cropW = Math.min(cropW, valid.width);
        cropH = Math.min(cropH, (int) Math.round(cropW / aspect));
        if (cropH > valid.height) {
            cropH = valid.height;
            cropW = Math.max(2, (int) Math.round(cropH * aspect));
            if (cropW > valid.width) {
                cropW = valid.width;
                cropH = Math.max(2, (int) Math.round(cropW / aspect));
            }
        }

        int x = valid.x + (valid.width - cropW) / 2;
        int y = valid.y + (int) Math.round((valid.height - cropH) * CROP_Y_BIAS);
        x = Math.max(0, Math.min(x, imgW - cropW));
        y = Math.max(0, Math.min(y, imgH - cropH));
        return new Rect(x, y, cropW, cropH);
    }

    static Rect aspectFit(Rect valid, int targetW, int targetH) {
        double aspect = (double) targetW / targetH;
        int w = valid.width;
        int h = valid.height;
        if ((double) w / h > aspect) {
            w = Math.max(2, (int) Math.round(h * aspect));
        } else {
            h = Math.max(2, (int) Math.round(w / aspect));
        }
        int x = valid.x + (valid.width - w) / 2;
        int y = valid.y + (valid.height - h) / 2;
        return new Rect(x, y, w, h);
    }

    static Rect clampRect(Rect r, int imgW, int imgH) {
        int x = Math.max(0, r.x);
        int y = Math.max(0, r.y);
        int w = r.width;
        int h = r.height;
        if (x + w > imgW) w = imgW - x;
        if (y + h > imgH) h = imgH - y;
        if (w < 1) w = 1;
        if (h < 1) h = 1;
        return new Rect(x, y, w, h);
    }

    /** Filled pixels only, inset so the circular fisheye rim is not stretched. */
    static Rect validPixelRect(Mat bgr) {
        int imgW = bgr.cols();
        int imgH = bgr.rows();
        Mat gray = new Mat();
        Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
        Mat mask = new Mat();
        Imgproc.threshold(gray, mask, VALID_LUMA_MIN, 255, Imgproc.THRESH_BINARY);

        int k = Math.max(7, Math.min(imgW, imgH) / 48);
        if ((k & 1) == 0) {
            k++;
        }
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(k, k));
        Imgproc.erode(mask, mask, kernel);

        Rect box = Imgproc.boundingRect(mask);
        gray.release();
        mask.release();
        kernel.release();

        if (box.width < imgW / 5 || box.height < imgH / 5) {
            int padX = (int) Math.round(imgW * 0.08);
            int padY = (int) Math.round(imgH * 0.10);
            return new Rect(padX, padY, Math.max(2, imgW - 2 * padX), Math.max(2, imgH - 2 * padY));
        }

        int insetX = Math.max(2, (int) Math.round(box.width * VALID_INSET_FRACTION));
        int insetY = Math.max(2, (int) Math.round(box.height * VALID_INSET_FRACTION));
        int x = box.x + insetX;
        int y = box.y + insetY;
        int w = box.width - 2 * insetX;
        int h = box.height - 2 * insetY;
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x + w > imgW) w = imgW - x;
        if (y + h > imgH) h = imgH - y;
        if (w < 2 || h < 2) {
            return new Rect(0, 0, imgW, imgH);
        }
        return new Rect(x, y, w, h);
    }

    static int estimateHorizonRow(Mat bgr) {
        Mat gray = new Mat();
        Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(9, 9), 1.4);
        Mat sobel = new Mat();
        Imgproc.Sobel(gray, sobel, CvType.CV_32F, 0, 1, 3);
        Mat mag = new Mat();
        Core.convertScaleAbs(sobel, mag);
        sobel.release();
        sobel = mag;

        int h = gray.rows();
        int w = gray.cols();
        int x0 = w / 6;
        int x1 = w - w / 6;
        int y0 = Math.max(1, (int) (h * 0.10));
        int y1 = Math.max(y0 + 1, (int) (h * 0.70));
        int fallback = (int) Math.round(HORIZON_FRACTION * h);
        int minValid = Math.max(8, (int) ((x1 - x0) * 0.40));

        double best = -1;
        int bestY = fallback;
        for (int y = y0; y < y1; y++) {
            Mat rowGray = gray.row(y).colRange(x0, x1);
            Mat valid = new Mat();
            Imgproc.threshold(rowGray, valid, VALID_LUMA_MIN, 1, Imgproc.THRESH_BINARY);
            int filled = Core.countNonZero(valid);
            valid.release();
            if (filled < minValid) {
                continue;
            }
            // Skip the black-to-content border created by rotation; the true
            // horizon has filled pixels both above and below.
            int yAbove = Math.max(0, y - 6);
            int yBelow = Math.min(h - 1, y + 6);
            Mat above = gray.row(yAbove).colRange(x0, x1);
            Mat below = gray.row(yBelow).colRange(x0, x1);
            Mat va = new Mat();
            Mat vb = new Mat();
            Imgproc.threshold(above, va, VALID_LUMA_MIN, 1, Imgproc.THRESH_BINARY);
            Imgproc.threshold(below, vb, VALID_LUMA_MIN, 1, Imgproc.THRESH_BINARY);
            int filledA = Core.countNonZero(va);
            int filledB = Core.countNonZero(vb);
            va.release();
            vb.release();
            if (filledA < minValid || filledB < minValid) {
                continue;
            }
            double s = Core.sumElems(sobel.row(y).colRange(x0, x1)).val[0];
            if (s > best) {
                best = s;
                bestY = y;
            }
        }
        gray.release();
        sobel.release();
        return bestY;
    }

    static double estimateRollDeg(Mat bgr) {
        Mat gray = new Mat();
        Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 1.0);
        Imgproc.Canny(gray, gray, 45, 140);
        Mat lines = new Mat();
        int minLen = Math.max(36, bgr.cols() / 7);
        Imgproc.HoughLinesP(gray, lines, 1, Math.PI / 180.0, 40, minLen, 14);
        ArrayList<double[]> scored = new ArrayList<>();
        for (int i = 0; i < lines.rows(); i++) {
            double[] l = lines.get(i, 0);
            if (l == null || l.length < 4) continue;
            double ang = Math.toDegrees(Math.atan2(l[3] - l[1], l[2] - l[0]));
            while (ang > 90)  ang -= 180;
            while (ang < -90) ang += 180;
            if (Math.abs(ang) > MAX_ROLL_DEG + 8) {
                continue;
            }
            double length = Math.hypot(l[2] - l[0], l[3] - l[1]);
            scored.add(new double[]{ ang, length });
        }
        gray.release();
        lines.release();
        if (scored.size() < 4) {
            return 0.0;
        }
        scored.sort((a, b) -> Double.compare(b[1], a[1]));
        int keep = Math.max(6, scored.size() / 2);
        if (keep > scored.size()) keep = scored.size();
        ArrayList<Double> angles = new ArrayList<>();
        for (int i = 0; i < keep; i++) {
            angles.add(scored.get(i)[0]);
        }
        java.util.Collections.sort(angles);
        double median = angles.get(angles.size() / 2);
        ArrayList<Double> clustered = new ArrayList<>();
        for (double a : angles) {
            if (Math.abs(a - median) <= 14.0) {
                clustered.add(a);
            }
        }
        if (clustered.size() >= 3) {
            java.util.Collections.sort(clustered);
            median = clustered.get(clustered.size() / 2);
        }
        if (Math.abs(median) < 1.5) {
            return 0.0;
        }
        return median;
    }

    static void forceExactSize(Mat img) {
        forceExactSize(img, TARGET_WIDTH, TARGET_HEIGHT);
    }

    static void forceExactSize(Mat img, int width, int height) {
        if (img.cols() == width && img.rows() == height) {
            return;
        }
        Mat tmp = new Mat();
        Imgproc.resize(img, tmp, new Size(width, height), 0, 0, Imgproc.INTER_AREA);
        tmp.copyTo(img);
        tmp.release();
    }


    //  CORE BLENDING  —  featherStitch

    static Mat featherStitch(Mat[] frames) {
        int N = frames.length;
        int H = frames[0].rows();
        int W = TARGET_WIDTH;

        int overlap = Math.min(OVERLAP_PX, W / 3);
        int panoW   = W + (N - 1) * (W - overlap);

        Mat accumColor  = Mat.zeros(H, panoW, CvType.CV_32FC3);
        Mat accumWeight = Mat.zeros(H, panoW, CvType.CV_32FC1);

        for (int i = 0; i < N; i++) {
            forceExactSize(frames[i], W, H);
            int xStart = i * (W - overlap);

            Mat weight = buildFeatherMask(H, W, overlap, i > 0, i < N - 1);
            multiplyByValidLuma(frames[i], weight);

            Mat frameF = new Mat();
            frames[i].convertTo(frameF, CvType.CV_32FC3);

            Mat weight3 = new Mat();
            List<Mat> ch = new ArrayList<>();
            ch.add(weight); ch.add(weight); ch.add(weight);
            Core.merge(ch, weight3);

            Mat wFrame = new Mat();
            Core.multiply(frameF, weight3, wFrame);

            int xEnd    = Math.min(xStart + W, panoW);
            int wActual = xEnd - xStart;

            Mat colorRoi  = accumColor.submat(0, H, xStart, xEnd);
            Mat weightRoi = accumWeight.submat(0, H, xStart, xEnd);

            Mat wFrameCrop = wFrame.colRange(0, wActual);
            Mat weightCrop = weight.colRange(0, wActual);

            Core.add(colorRoi,  wFrameCrop, colorRoi);
            Core.add(weightRoi, weightCrop, weightRoi);

            colorRoi.release(); weightRoi.release();
            frameF.release(); weight.release(); weight3.release();
            wFrame.release();
        }

        Mat safeW = new Mat();
        Core.max(accumWeight, new Scalar(1e-6), safeW);

        Mat safeW3 = new Mat();
        List<Mat> wch = new ArrayList<>();
        wch.add(safeW); wch.add(safeW); wch.add(safeW);
        Core.merge(wch, safeW3);

        Mat blended = new Mat();
        Core.divide(accumColor, safeW3, blended);

        Mat result = new Mat();
        blended.convertTo(result, CvType.CV_8UC3);

        accumColor.release(); accumWeight.release();
        safeW.release(); safeW3.release(); blended.release();

        return result;
    }

    static Mat buildFeatherMask(int H, int W, int overlap, boolean fadeLeft, boolean fadeRight) {
        Mat mask = new Mat(H, W, CvType.CV_32FC1, new Scalar(1.0));
        if (overlap <= 0) {
            return mask;
        }
        for (int x = 0; x < overlap; x++) {
            float alpha = (float) (0.5 - 0.5 * Math.cos(Math.PI * x / overlap));
            if (fadeLeft) {
                Mat colL = mask.col(x);
                colL.setTo(new Scalar(alpha));
                colL.release();
            }
            if (fadeRight) {
                Mat colR = mask.col(W - 1 - x);
                colR.setTo(new Scalar(alpha));
                colR.release();
            }
        }
        return mask;
    }

    /** Keep black letterbox from contributing to the blend (avoids dark seams). */
    static void multiplyByValidLuma(Mat bgr, Mat weight) {
        Mat gray = new Mat();
        Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
        Mat valid = new Mat();
        Imgproc.threshold(gray, valid, VALID_LUMA_MIN, 1.0, Imgproc.THRESH_BINARY);
        Mat validF = new Mat();
        valid.convertTo(validF, CvType.CV_32FC1);
        Core.multiply(weight, validF, weight);
        gray.release();
        valid.release();
        validF.release();
    }

    /**
     * Residual vertical registration in the overlap so objects that straddle a
     * seam (e.g. the yellow car between right and rear) sit on one line.
     */
    static double[] estimatePairwiseDy(Mat[] frames) {
        int N = frames.length;
        double[] dy = new double[N];
        int overlap = Math.min(OVERLAP_PX, TARGET_WIDTH / 3);
        double maxStep = TARGET_HEIGHT * 0.22;
        for (int i = 0; i < N - 1; i++) {
            Point shift = overlapShift(frames[i], frames[i + 1], overlap);
            if (shift != null && Math.abs(shift.y) <= maxStep) {
                dy[i + 1] = dy[i] - shift.y;
            } else {
                dy[i + 1] = dy[i];
            }
        }
        double mean = 0;
        for (double v : dy) mean += v;
        mean /= N;
        for (int i = 0; i < N; i++) {
            dy[i] -= mean;
        }
        return dy;
    }

    static Point overlapShift(Mat left, Mat right, int overlap) {
        if (left.empty() || right.empty() || overlap < 16) {
            return null;
        }
        int H = Math.min(left.rows(), right.rows());
        int W = Math.min(left.cols(), right.cols());
        int y0 = (int) (H * 0.18);
        int y1 = (int) (H * 0.82);
        if (y1 - y0 < 24 || W < overlap) {
            return null;
        }
        Mat a = left.submat(y0, y1, W - overlap, W);
        Mat b = right.submat(y0, y1, 0, overlap);
        Mat ga = new Mat();
        Mat gb = new Mat();
        Imgproc.cvtColor(a, ga, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(b, gb, Imgproc.COLOR_BGR2GRAY);
        Mat fa = new Mat();
        Mat fb = new Mat();
        ga.convertTo(fa, CvType.CV_32FC1);
        gb.convertTo(fb, CvType.CV_32FC1);
        if (Core.mean(fa).val[0] < 18 || Core.mean(fb).val[0] < 18) {
            a.release(); b.release();
            ga.release(); gb.release();
            fa.release(); fb.release();
            return null;
        }
        Mat win = new Mat();
        Imgproc.createHanningWindow(win, fa.size(), CvType.CV_32FC1);
        Point shift = Core.phaseCorrelate(fa, fb, win);
        a.release(); b.release();
        ga.release(); gb.release();
        fa.release(); fb.release();
        win.release();
        return shift;
    }

    static Mat[] applyVerticalShifts(Mat[] frames, double[] dy) {
        if (dy == null) {
            return frames;
        }
        boolean any = false;
        for (double v : dy) {
            if (Math.abs(v) > 0.5) {
                any = true;
                break;
            }
        }
        if (!any) {
            return frames;
        }
        Mat[] out = new Mat[frames.length];
        for (int i = 0; i < frames.length; i++) {
            Mat M = Mat.zeros(2, 3, CvType.CV_64FC1);
            M.put(0, 0, 1);
            M.put(1, 1, 1);
            M.put(1, 2, dy[i]);
            out[i] = new Mat();
            Imgproc.warpAffine(frames[i], out[i], M,
                    new Size(frames[i].cols(), frames[i].rows()),
                    Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
            M.release();
        }
        return out;
    }


    // VIDEO IO  —  FFmpeg plugin, then MSMF without RGB32 conversion

    static Path preferH264(Path requested) {
        return ensureDecodable(requested);
    }

    /**
     * These camera .mov files are PNG video (FFmpeg codec_id=61, fourcc png).
     * OpenCV's bundled FFmpeg and Windows MSMF cannot decode that.
     * Use a sibling H.264 .mp4, transcoding with ffmpeg when needed.
     */
    static Path ensureDecodable(Path requested) {
        Path mp4 = siblingWithExt(requested, ".mp4");
        if (isUsableFile(mp4)) {
            System.out.println("Using " + mp4.getFileName() + " (H.264) instead of "
                    + requested.getFileName());
            return mp4;
        }
        if (!isUsableFile(requested)) {
            return requested;
        }
        if (transcodeToH264(requested, mp4) && isUsableFile(mp4)) {
            return mp4;
        }
        System.err.println("Cannot decode " + requested.getFileName()
                + " (PNG-in-MOV). Install ffmpeg on PATH and re-run, or convert:");
        System.err.println("  ffmpeg -y -i " + requested.getFileName()
                + " -c:v libx264 -pix_fmt yuv420p -an " + mp4.getFileName());
        return requested;
    }

    static Path siblingWithExt(Path requested, String ext) {
        String name = requested.getFileName().toString();
        int dot = name.lastIndexOf('.');
        String stem = dot >= 0 ? name.substring(0, dot) : name;
        Path dir = requested.toAbsolutePath().getParent();
        if (dir == null) {
            dir = Paths.get(".");
        }
        return dir.resolve(stem + ext);
    }

    static boolean isUsableFile(Path path) {
        try {
            return path != null && Files.isRegularFile(path) && Files.size(path) > 0;
        } catch (IOException e) {
            return false;
        }
    }

    static boolean transcodeToH264(Path src, Path dst) {
        System.out.println("Transcoding " + src.getFileName() + " -> " + dst.getFileName()
                + " (PNG MOV cannot be decoded by OpenCV FFmpeg/MSMF)");
        ProcessBuilder pb = new ProcessBuilder(
                "ffmpeg", "-hide_banner", "-y",
                "-i", src.toAbsolutePath().toString(),
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-an", dst.toAbsolutePath().toString());
        pb.inheritIO();
        try {
            int code = pb.start().waitFor();
            if (code == 0 && isUsableFile(dst)) {
                System.out.println("Transcode OK: " + dst.getFileName());
                return true;
            }
            System.err.println("ffmpeg exited with code " + code);
        } catch (IOException e) {
            System.err.println("ffmpeg not found on PATH: " + e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("ffmpeg transcode interrupted");
        }
        try {
            Files.deleteIfExists(dst);
        } catch (IOException ignored) {
        }
        return false;
    }

    /**
     * Official Windows OpenCV puts opencv_java*.dll in build/java/x64 and the
     * FFmpeg videoio plugin in build/bin. java.library.path usually only has
     * the first folder, so MSMF is used and .mov RGB32 decode fails.
     */
    static void loadFfmpegPlugin() {
        String[] dllNames = {
            "opencv_videoio_ffmpeg490_64.dll",
            "opencv_videoio_ffmpeg4.dll",
            "opencv_videoio_ffmpeg.dll"
        };
        List<Path> dirs = new ArrayList<>();
        String libPath = System.getProperty("java.library.path", "");
        for (String dir : libPath.split(File.pathSeparator)) {
            if (dir == null || dir.trim().isEmpty()) continue;
            Path p = Paths.get(dir.trim()).toAbsolutePath().normalize();
            dirs.add(p);
            if (p.getParent() != null) {
                dirs.add(p.getParent());
                if (p.getParent().getParent() != null) {
                    dirs.add(p.getParent().getParent().resolve("bin"));
                }
            }
        }
        dirs.add(Paths.get("opencv", "build", "bin").toAbsolutePath());
        dirs.add(Paths.get("..", "opencv", "build", "bin").toAbsolutePath());

        for (Path dir : dirs) {
            for (String dll : dllNames) {
                Path candidate = dir.resolve(dll).normalize();
                if (!Files.isRegularFile(candidate)) continue;
                try {
                    System.load(candidate.toString());
                    System.out.println("Loaded FFmpeg videoio plugin: " + candidate);
                    return;
                } catch (Throwable t) {
                    System.err.println("Could not load " + candidate + ": " + t.getMessage());
                }
            }
        }
        System.err.println("FFmpeg videoio plugin not loaded. MSMF will be used for .mov "
                + "and may fail RGB32. Copy opencv_videoio_ffmpeg490_64.dll next to "
                + "opencv_java490.dll or into opencv\\build\\bin on PATH.");
    }

    static VideoCapture openVideo(Path path) {
        String file = path.toAbsolutePath().toString();

        int[] apis = {
            Videoio.CAP_FFMPEG,
            Videoio.CAP_ANY,
            Videoio.CAP_MSMF
        };
        String[] labels = { "FFMPEG", "ANY", "MSMF" };

        for (int a = 0; a < apis.length; a++) {
            for (double convert : new double[]{ (apis[a] == Videoio.CAP_MSMF ? 0 : 1), 0, 1 }) {
                VideoCapture cap = tryOpen(file, apis[a], labels[a], convert);
                if (cap != null) {
                    return cap;
                }
            }
        }

        VideoCapture cap = new VideoCapture();
        cap.open(file);
        if (cap.isOpened()) {
            cap.set(Videoio.CAP_PROP_CONVERT_RGB, 0);
            if (probeFrame(cap)) {
                logBackend(file, cap, "default");
                return cap;
            }
        }
        cap.release();
        return null;
    }

    static VideoCapture tryOpen(String file, int api, String label, double convertRgb) {
        VideoCapture cap = new VideoCapture();
        MatOfInt params = new MatOfInt(Videoio.CAP_PROP_CONVERT_RGB, (int) convertRgb);
        try {
            boolean opened = cap.open(file, api, params);
            if (!opened || !cap.isOpened()) {
                cap.release();
                params.release();
                return null;
            }
        } catch (Exception e) {
            cap.release();
            params.release();
            return null;
        }
        params.release();

        cap.set(Videoio.CAP_PROP_CONVERT_RGB, convertRgb);
        if (!probeFrame(cap)) {
            cap.release();
            return null;
        }
        logBackend(file, cap, label + " convertRGB=" + (int) convertRgb);
        return cap;
    }

    static boolean probeFrame(VideoCapture cap) {
        Mat probe = new Mat();
        boolean ok = cap.read(probe) && !probe.empty();
        if (ok) {
            Mat bgr = new Mat();
            ok = toBgr(probe, bgr);
            bgr.release();
        }
        probe.release();
        if (!ok) {
            return false;
        }
        cap.set(Videoio.CAP_PROP_POS_FRAMES, 0);
        cap.set(Videoio.CAP_PROP_POS_MSEC, 0);
        return true;
    }

    static void logBackend(String file, VideoCapture cap, String requested) {
        String backend = requested;
        try {
            String named = cap.getBackendName();
            if (named != null && !named.isEmpty()) {
                backend = named + " / " + requested;
            }
        } catch (Exception ignored) {
        }
        System.out.println("Opened " + file + " [" + backend + "]");
    }

    static boolean toBgr(Mat src, Mat dst) {
        if (src == null || src.empty()) {
            return false;
        }
        int type = src.type();
        if (type == CvType.CV_8UC3) {
            src.copyTo(dst);
            return true;
        }
        if (type == CvType.CV_8UC4) {
            Imgproc.cvtColor(src, dst, Imgproc.COLOR_BGRA2BGR);
            return !dst.empty();
        }
        if (type == CvType.CV_8UC2) {
            Imgproc.cvtColor(src, dst, Imgproc.COLOR_YUV2BGR_YUY2);
            return !dst.empty();
        }
        if (src.channels() == 1) {
            int h = src.rows();
            int w = src.cols();
            if (h * 2 % 3 == 0) {
                int visH = h * 2 / 3;
                if (visH > 0 && visH * 3 / 2 == h) {
                    try {
                        Imgproc.cvtColor(src, dst, Imgproc.COLOR_YUV2BGR_NV12);
                        if (!dst.empty() && dst.channels() == 3) {
                            return true;
                        }
                    } catch (Exception ignored) {
                    }
                }
            }
            Imgproc.cvtColor(src, dst, Imgproc.COLOR_GRAY2BGR);
            return !dst.empty();
        }
        src.copyTo(dst);
        return !dst.empty();
    }

    static boolean readBgr(VideoCapture cap, Mat bgr) {
        if (cap == null || !cap.isOpened()) {
            return false;
        }
        Mat raw = new Mat();
        if (!cap.read(raw) || raw.empty()) {
            raw.release();
            return false;
        }
        boolean ok = toBgr(raw, bgr);
        raw.release();
        return ok && bgr != null && !bgr.empty();
    }

    static boolean readOrLoop(VideoCapture[] caps, int i, Path path, Mat frame) {
        if (readBgr(caps[i], frame)) {
            return true;
        }

        caps[i].set(Videoio.CAP_PROP_POS_FRAMES, 0);
        caps[i].set(Videoio.CAP_PROP_POS_MSEC, 0);
        if (readBgr(caps[i], frame)) {
            System.out.println("Video " + i + " looped via seek");
            return true;
        }

        caps[i].release();
        caps[i] = openVideo(path);
        if (readBgr(caps[i], frame)) {
            System.out.println("Video " + i + " reopened after loop");
            return true;
        }

        System.err.println("Warning: Could not read frame from video " + i);
        return false;
    }

    static byte[] encodeJpeg(Mat frame) {
        MatOfByte buf    = new MatOfByte();
        MatOfInt  params = new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 88);
        Imgcodecs.imencode(".jpg", frame, buf, params);
        return buf.toArray();
    }

    static void writeFrame(OutputStream out, byte[] jpeg) throws IOException {
        String header = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                      + jpeg.length + "\r\n\r\n";
        out.write(header.getBytes("UTF-8"));
        out.write(jpeg);
        out.write("\r\n".getBytes("UTF-8"));
        out.flush();
    }
}
