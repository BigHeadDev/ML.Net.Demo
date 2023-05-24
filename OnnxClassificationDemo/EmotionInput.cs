using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxClassificationDemo {
    public class EmotionInput {
        [ImageType(64, 64)]
        public MLImage Image { get; set; }
    }
}
