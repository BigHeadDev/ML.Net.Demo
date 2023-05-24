using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxClassificationDemo {
    public class EmotionPrediction {
        private readonly string modelFile = "emotion-ferplus-8.onnx";
        private string[] emotions = new string[] { "一般", "快乐", "惊讶", "伤心", "生气", "疑惑", "害怕", "蔑视" };
        private PredictionEngine<EmotionInput, EmotionOutput> predictionEngine;
        public EmotionPrediction()
        {
            MLContext context = new MLContext();
            var emptyData = new List<EmotionInput>();
            var data = context.Data.LoadFromEnumerable(emptyData);
            var pipeline = context.Transforms.ResizeImages("resize", 64, 64, inputColumnName: nameof(EmotionInput.Image), Microsoft.ML.Transforms.Image.ImageResizingEstimator.ResizingKind.Fill).
                Append(context.Transforms.ExtractPixels("Input3", "resize", Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator.ColorBits.Blue)).
                Append(context.Transforms.ApplyOnnxModel(modelFile));

            var model = pipeline.Fit(data);
            predictionEngine = context.Model.CreatePredictionEngine<EmotionInput, EmotionOutput>(model);    
        }

        public string Predict(string path) {
            using (var stream = new FileStream(path, FileMode.Open)) {
                using(var bitmap = MLImage.CreateFromStream(stream)) {
                    var result = predictionEngine.Predict(new EmotionInput() { Image = bitmap });
                    var max = result.Result.Max();
                    var index = result.Result.ToList().IndexOf(max);
                    return emotions[index];
                }
            }
        }
    }
}
