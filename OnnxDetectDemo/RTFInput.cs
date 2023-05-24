using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxDetectDemo {
    public class RTFInput {
        [ImageType(640, 480)]
        public MLImage Image { get; set; }
    }
}
