using Microsoft.ML.Data;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using System.Windows;

namespace OnnxDetectDemo {
    public class RTFPrediction {
        private readonly string modelFile = "version-RFB-640.onnx";
        private PredictionEngine<RTFInput, RTFOutput> predictionEngine = null;
        public RTFPrediction() {
            MLContext context = new MLContext();
            var emptyData = new List<RTFInput>();
            var data = context.Data.LoadFromEnumerable(emptyData);
            var pipeline =
                  context.Transforms.ResizeImages(
                  resizing: Microsoft.ML.Transforms.Image.ImageResizingEstimator.ResizingKind.Fill,//填充Resize
                  outputColumnName: "resize",//Resize的结果放置到 data列
                  imageWidth: 640,
                  imageHeight: 480,
                  inputColumnName: nameof(RTFInput.Image)//从Image属性来源,
                  )
                  .Append(
                      context.Transforms.ExtractPixels(
                          offsetImage: 127f,
                          scaleImage: 1 / 128f,
                          inputColumnName: "resize",
                          outputColumnName: "input")
                  ).Append(
                  context.Transforms.ApplyOnnxModel(
                      modelFile: modelFile,
                      inputColumnNames: new string[] { "input" },
                      outputColumnNames: new string[] { "scores", "boxes" }));
            var model = pipeline.Fit(data);
            predictionEngine = context.Model.CreatePredictionEngine<RTFInput, RTFOutput>(model);//生成预测引擎
        }

        public ImageSource Predict(string path) {
            using (var stream = new FileStream(path, FileMode.Open)) {
                using (var bitmap = MLImage.CreateFromStream(stream)) {
                    var prediction = predictionEngine.Predict(new RTFInput() { Image = bitmap });
                    var boxes = ParseBox(prediction);
                    boxes = boxes.Where(b => b.Score > 0.9).OrderByDescending(s => s.Score).ToList();
                    boxes = HardNMS(boxes, 0.4);
                    var bitmapimage = new BitmapImage(new Uri(path));
                    var rtb = new RenderTargetBitmap(bitmap.Width, bitmap.Height, 96, 96, PixelFormats.Pbgra32);
                    int t = 0;

                    var dv = new DrawingVisual();
                    using (DrawingContext dc = dv.RenderOpen()) {
                        dc.DrawImage(bitmapimage, new Rect(0, 0, bitmap.Width, bitmap.Height));
                        foreach (var item in boxes) {
                            dc.DrawRectangle(null, new Pen(Brushes.Red, 2), new Rect(item.Rect.X * bitmap.Width, item.Rect.Y * bitmap.Height, item.Rect.Width * bitmap.Width, item.Rect.Height * bitmap.Height));
                        }
                    }
                    rtb.Render(dv);
                    return rtb;

                }

            }
        }

        private List<Box> ParseBox(RTFOutput prediction) {
            var length = prediction.Boxes.Length / 4;
            var boxes = Enumerable.Range(0, length).Select(i => new Box() {
                X1 = prediction.Boxes[i * 4],
                Y1 = prediction.Boxes[i * 4 + 1],
                X2 = prediction.Boxes[i * 4 + 2],
                Y2 = prediction.Boxes[i * 4 + 3],
                Score = prediction.Scores[i * 2 + 1]
            }
            );
            boxes = boxes.OrderByDescending(b => b.Score);
            return boxes.ToList();
        }

        public List<Box> HardNMS(List<Box> boxes, double overlapThreshold) {

            var selectedBoxes = new List<Box>();
            while (boxes.Count > 0) {
                // 取出置信度最高的bbox
                var currentBox = boxes[0];
                selectedBoxes.Add(currentBox);

                // 计算当前bbox和其余bbox之间的IOU
                boxes.RemoveAt(0);
                for (int i = boxes.Count - 1; i >= 0; i--) {
                    var iou = CalculateIOU(currentBox, boxes[i]);
                    if (iou >= overlapThreshold) {
                        boxes.RemoveAt(i);
                    }
                }
            }
            return selectedBoxes;
        }
        public double CalculateIOU(Box boxA, Box boxB) {
            // 计算相交部分的坐标信息
            float xOverlap = Math.Max(0, Math.Min(boxA.X2, boxB.X2) - Math.Max(boxA.X1, boxB.X1) + 1);
            float yOverlap = Math.Max(0, Math.Min(boxA.Y2, boxB.Y2) - Math.Max(boxA.Y1, boxB.Y1) + 1);

            // 计算相交部分的面积和并集部分的面积
            float intersectionArea = xOverlap * yOverlap;
            float unionArea = boxA.Area + boxB.Area - intersectionArea;

            // 计算IoU
            double iou = (double)intersectionArea / unionArea;
            return iou;
        }
    }

    public class Box {
        public float X1 { get; set; }
        public float Y1 { get; set; }
        public float X2 { get; set; }
        public float Y2 { get; set; }
        public float Score { get; set; }
        // 计算面积
        public float Area => (X2 - X1 + 1) * (Y2 - Y1 + 1);

        private Rect GetRect() {
            var hei = Y2 - Y1;
            var wid = X2 - X1;
            if (wid < 0) {
                wid = 0;
            }
            if (hei < 0) {
                hei = 0;
            }
            return new Rect(X1, Y1, wid, hei);
        }

        private Rect rect = Rect.Empty;

        public Rect Rect {
            get {
                if (rect.IsEmpty) {
                    rect = GetRect();
                }
                return rect;
            }
        }

    }
}
