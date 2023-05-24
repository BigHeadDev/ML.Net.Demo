using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxClassificationDemo {
    public class EmotionOutput {
        [ColumnName("Plus692_Output_0")]
        public float[] Result { get; set; }
    }
}
