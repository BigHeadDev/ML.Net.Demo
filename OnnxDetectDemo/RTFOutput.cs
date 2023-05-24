using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxDetectDemo {
    public class RTFOutput {
        [ColumnName("scores")]
        [VectorType(1, 17640, 2)]
        public float[] Scores { get; set; }
        [ColumnName("boxes")]
        [VectorType(1, 17640, 4)]
        public float[] Boxes { get; set; }
    }
}
