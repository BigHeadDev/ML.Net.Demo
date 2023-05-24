> ML.Net - 开源的跨平台机器学习框架
> - 支持CPU/GPU训练
> - 轻松简洁的预测代码
> - 可扩展其他的机器学习平台
> - 跨平台

# 1.使用Visual Studio的Model Builder训练和使用模型
> Visual Studio默认安装了Model Builder插件，可以很快地进行一些通用模型类型的训练和部署，提高接入机器学习的开发效率

## 1.1 新建模型
通过非常简单地 右键项目-添加-机器学习模型

![img](https://img2023.cnblogs.com/blog/1339560/202305/1339560-20230524165510435-1813123810.png)

## 1.2 选择模型
ModelBuilder中提供了集中常用的模型类型以供开发者使用，开发者可以通过这些类别的模型快速接入，并且训练自己的数据，本节内容将会使用计算机视觉中的<b>”图像分类“</b>进行演示

![img](https://img2023.cnblogs.com/blog/1339560/202305/1339560-20230524165747536-1739480461.png)

## 1.3 选择训练环境
接下来要选择训练的环境，提供了CPU/GPU/Azure云三种方式训练，这里为了简单演示，我使用了CPU训练，如果数据量大且复杂的请选择GPU，并且提前安装CUDA、cuDNN
![img](https://img2023.cnblogs.com/blog/1339560/202305/1339560-20230524170406729-1992135999.png)

## 1.4 添加训练数据
我从搜索引擎中，搜集到了一系列”奥特曼“的图片（我相信不是所有人都可以认出各个时代的各个奥特曼 哈哈哈）
然后将这些图片进行了文件夹分类，导入到ModelBuilder中，如下：
![img](https://img2023.cnblogs.com/blog/1339560/202305/1339560-20230524170617306-2135417598.png)

## 1.5 开始训练
本次演示训练157张图片，耗时50秒
![img](https://img2023.cnblogs.com/blog/1339560/202305/1339560-20230524170711373-2085576454.png)

## 1.6 评估
此环节，为了检验训练成果和准确率，ModelBuilder中提供了图形化的方式进行预测检测，我在另外的搜索引擎中，<b>找到了一张没有经过训练的图片</b>，它准确地判断出了”迪迦奥特曼“的概率为63%
![img](https://img2023.cnblogs.com/blog/1339560/202305/1339560-20230524170930534-1377099317.png)

## 1.7 代码编写
这一环节中，ModelBuilder给出了示例代码，直接复制粘贴就可以用到自己的实际项目中
同时还提供了，一键生成控制台或者WebAPI项目的入口。给力！🙌

我新建了一个WPF项目，添加了一个Button，进行简单测试：
```Xml
 <Grid>
        <Button Click="Button_Click" Content="预测一个奥特曼" />
    </Grid>
```
```C#
private void Button_Click(object sender, RoutedEventArgs e) {
            OpenFileDialog dialog= new OpenFileDialog();
            if (dialog.ShowDialog().Value) {
                //Load sample data
                var imageBytes = File.ReadAllBytes(dialog.FileName);
                UltraMan.ModelInput sampleData = new UltraMan.ModelInput() {
                    ImageSource = imageBytes,
                };

                //Load model and predict output
                var result = UltraMan.Predict(sampleData);
                if (result.Score.Any(s=>s>=0.6)) {
                    MessageBox.Show(result.PredictedLabel);
                } else {
                    MessageBox.Show("没有识别到奥特曼");
                }
            }
            
        }
```
选择一张图片，导入之后，即可弹出预测结果
![img](https://img2023.cnblogs.com/blog/1339560/202305/1339560-20230524171359160-223125995.png)

> ModelBuilder的操作非常简单，基本不需要了解机器学习的原理或者python，对一些有这些内置模型需求的.Net开发者很有帮助！~🥳

---

# 2.使用ONNX模型进行分类预测
`如果团队中有其他专业的AI人员进行模型训练和机器学习代码编写，如何将pytorch、tensenflow等框架训练的模型用在.Net中呢？`
> ML.Net在支持使用内置的ModelBuilder模型外，还支持使用onnx模型进行预测

> - [开放式神经网络交换 （ONNX）](https://onnx.ai/) 是一种用于表示机器学习模型的开放标准格式。ONNX 由合作伙伴社区提供支持，这些合作伙伴已在许多框架和工具中实现了它
> - 开源的onnx模型系列下载（有很多第三方的优质onnx现成模型可供下载使用）：[onnx/models：ONNX 格式的预训练、最先进的模型集合 (github.com)](https://github.com/onnx/models)
> - ML模型仪表盘(可以查看模型详细推导流程和输入输出列) ：[Release WinML Dashboard v0.7.0 · microsoft/Windows-Machine-Learning (github.com)](https://github.com/Microsoft/Windows-Machine-Learning/releases/tag/v0.7.0)

这里需要提前介绍一下ML模型仪表盘
![img](https://img2023.cnblogs.com/blog/1339560/202305/1339560-20230524172118398-1112776029.png)
<b>右侧的Inputs和Outputs在后续步骤中比较关键</b>

## 2.1 下载所需模型
进入github根据需求下载模型文件，本篇文章使用了<b>【Emotion FERPlus】</b>模型进行情绪预测
下载地址：[github](https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus)


根据github中的接入说明，理解输入、输出、预处理、预测、后处理等流程后，开始接入
![img](https://img2023.cnblogs.com/blog/1339560/202305/1339560-20230524172511924-1681483160.png)

## 2.2 流程说明

>输入：N\*1\*64\*64 的float数组

表示可以预测多张(N)图片，且输入图片要是单色通道图(1)，尺寸为64\*64（需要缩放）

>预处理：导入图片路径进行预测

python代码中，将图片导入，并进行了缩放处理，然后使用np.array把图片数据转为了float数组形式，最后把数组进行\[1,1,64,64\]的形状缩放，将rgb提取了单色数据

>输出：1\*8 的float数组

输出了一个8长度的一维数组，分别代表了8种表情的分数值，可能性最高的值为最终结果


## 2.3 ML.Net接入

>使用ML.Net接入onnx前，需要安装几个nuget包：
> - Microsoft.ML
> - Microsoft.ML.ImageAnalytics
> - Microsoft.ML.OnnxTransformer

### 2.3.1 定义输入和输出的类

📢输入：
```C#
    public class EmotionInput {
            [ImageType(64,64)]
            public MLImage Image { get; set; }  
        }
```
定义了一个输入类EmotionInput，标记图像为64\*64，且类型为MLImage

📢输出：
```C#
    public class EmotionOutput {
            [ColumnName("Plus692_Output_0")]
            public float[] Result { get; set; }
        }
```
根据ML Dashboard可以看到输出列名为Plus692\_Output\_0，类型为一维浮点数组

📢开始预测：
```C#
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
```
`其中预测部分，比较关键的地方是预测管道部分，从Input中拿到图片数据-->Resize-->提取图片的蓝色数据-->作为Input3输入列传入模型` 

`这里使用蓝色作为提取色，是因为蓝色在色彩表示中较为明亮，计算机更容易识别这些像素和区域`

📢使用WPF接入试试看：
```xml
<Grid>
        <Button Click="Button_Click" Content="预测表情" />
    </Grid>
```
```C#
private EmotionPrediction prediction = new EmotionPrediction();

        private void Button_Click(object sender, RoutedEventArgs e) {
            OpenFileDialog dialog = new OpenFileDialog();
            if (dialog.ShowDialog().Value) {
                var result = prediction.Predict(dialog.FileName);
                MessageBox.Show(result);
            }
        }
```
导入一张普通图片试试：
![img](https://img2023.cnblogs.com/blog/1339560/202305/1339560-20230524173557096-1576019552.jpg)
![img](https://img2023.cnblogs.com/blog/1339560/202305/1339560-20230524173638931-1733676848.png)

<font color="pink">导入一张甜心美少女试试：</font>
![img](https://img2023.cnblogs.com/blog/1339560/202305/1339560-20230524173714830-158726017.jpg)
![img](https://img2023.cnblogs.com/blog/1339560/202305/1339560-20230524173732244-1035615384.png)

onnx模型的接入，使得ML.Net的可扩展性更高，不仅仅是内置模型，还可以更多~

---

## 3. 使用ONNX模型进行识别分割
> 接下来是一个稍微复杂一点的模型的接入方法  
> 卷积神经网络中，人脸识别、车牌识别、物体识别很火热（比如有名的开源模型YOLO）  
> 当然ModelBuilder已经内置物体识别模型，可以识别的物体在图中位置和矩形区域

进入github根据需求下载模型文件，本篇文章使用了<b>【UltraFace】</b>模型进行人脸识别  
下载地址:[github](https://github.com/onnx/models/tree/main/vision/body_analysis/ultraface)  

![img](https://img2023.cnblogs.com/blog/1339560/202305/1339560-20230524174402465-1381575758.png)

通过描述，可以看出此模型会复杂一些。需要将数据进行BGR2RGB、偏移、归一化等预处理，也需要对预测结果进行非极大值抑制、矩形框变化处理

## 3.1 流程说明

🔔输入：1\*3\*320\*240

一张图片，进行缩放处理到320\*240，且不要透明度

🔔预处理：BRG-RGB + Resize + 偏移 + 归一化 + 数据转chw

这里是opencv中的一系列图片处理的方法，为了匹配模型的输入且加快训练预测的速度


> 在深度学习中，神经网络模型的输入数据一般都需要经过一些预处理才能被正确地输入到模型中进行训练或者预测。下面是对各个预处理步骤的解释：

> \- BGR-RGB转换：在OpenCV中读取图像时，图像的通道顺序是BGR，而在深度学习中通常使用的是RGB格式。因此，需要对输入图像进行BGR-RGB通道转换。

> \- Resize：由于神经网络模型对输入图像的大小有一定的要求，因此在输入图像大小不符合要求时，需要进行图像的缩放操作。缩放操作有助于保留输入图像中的重要特征，并且可以减少训练和预测的时间和计算资源消耗。

> \- 偏移：在进行归一化操作前，先将图像每个像素点的值减去一个常数，这个常数一般是对训练数据集像素值取平均值。通过这个操作，可以将输入图像的像素值整体向左偏移一定的偏移量，使得整个像素值的范围更加平衡，便于模型的训练和优化。

> \- 归一化：在神经网络模型中，通过对输入数据进行归一化的操作，可以使得数据更加平滑，减少噪声和异常情况的影响。一般地，归一化会将数据的数值范围缩放到0到1之间或者-1到1之间（或其他固定范围内），这样有助于加快训练和提高模型的稳定性。

> \- 数据转置和重排：在深度学习框架中，输入数据的格式通常是(batch\_size, channel, height, width)，所以需要将预处理后的图像从(height, width, channel)的格式转化为(channel, height, width)的格式，并加上一维batch\_size，以便于输入到网络中进行训练或者预测。

> 综上所述，这些预处理步骤是为了将图像处理成与模型输入相对应的格式，并且预处理后的图像可以减少噪声、保留重要特征、加快训练和提高模型的稳定性。

🔔输出：1\*4420\*2 和 1\*4420\*4的两个数组

分别代表了分数和矩形框的数据

🔔后处理：矩形框的转换+非极大值抑制

需要对输出的两个矩形进行处理，根据分数排列、根据非极大值抑制筛选出最确定的矩形框结果


> 在目标检测中，经常会出现多个检测框（bounding box）重叠覆盖同一目标的情况，而我们通常只需要保留一个最佳的检测结果。非极大值抑制（Non-Maximum Suppression，NMS）就是一种常见的目标检测算法，用于在冗余的检测框中筛选出最佳的一个。

> NMS 原理是在对检测结果进行处理前，按照检测得分进行排序（一般检测得分越高，表明检测框越可能包含目标），然后选择得分最高的检测框加入结果中。接下来，遍历排序后的其余检测框，如果检测框之间的IoU（Intersection over Union，交并比）大于一定阈值，那么就将该检测框删除，因为被保留的那一个框已经足够表明目标的存在。

> 该过程不断迭代，直到所有框都被遍历完毕为止，从未删除的框中即为最终结果。由于 NMS 算法可以过滤掉重叠检测框中的冗余结果，因此在很多基于深度学习的目标检测算法（如 YOLO、SSD 等）中都被广泛使用。


== 以上作为了解，具体看下面代码 ==

## 3.2 ML.Net接入

### 3.2.1 输入
```C#
    public class RTFInput {
            [ImageType(640, 480)]
            public MLImage Image { get; set; }
        }
```
### 3.2.2 输出
```C#
    public class RTFOutput {
            [ColumnName("scores")]
            [VectorType(1, 17640, 2)]
            public float[] Scores { get; set; }
            [ColumnName("boxes")]
            [VectorType(1, 17640, 4)]
            public float[] Boxes { get; set; }
        }
```
### 3.2.3 预测
```C#
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
    
```
Box的定义：
```C#
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
```
### 3.2.4 使用WPF试试看~
```xml
<Grid>
        <Grid.RowDefinitions>
            <RowDefinition Height="50" />
            <RowDefinition />
        </Grid.RowDefinitions>
        <Button Click="Button_Click" Content="识别人脸框" />
        <Image x:Name="image" Grid.Row="1" />
    </Grid>
```
```C#
private RTFPrediction prediction = new RTFPrediction();

        private void Button_Click(object sender, RoutedEventArgs e) {
            OpenFileDialog dialog = new OpenFileDialog();
            if (dialog.ShowDialog().Value) {
                var result = prediction.Predict(dialog.FileName);
                if (result!=null) {
                    image.Source = result;
                }
            }
        }
```
<font color="pink">还是导入一张快乐美少女~</font>

![img](https://img2023.cnblogs.com/blog/1339560/202305/1339560-20230524175120309-606453931.png)

<font color="pink">识别到了！！并且我在WPF中用DrawingContext为它绘制了红色矩形框。哎呀！应该用粉色！</font>



再导入一张多人脸图试试？
![img](https://img2023.cnblogs.com/blog/1339560/202305/1339560-20230524175757271-1767704390.png)
完美~~

# 4. 其他

本文只描述了三种训练或者接入机器学习的方式，应该可以实现一部分机器学习的需求，本文只是作为一个.NET平台机器学习预测的例子，并没有将ML.Net和传统Pytorch等成熟框架进行比较，只是给出另外一种选择。

作为绝大多数.Net开发者，如果有现成的、不需要跨语言、上手成本低的机器学习框架可以用在现有业务中，当然没有必要再学习专业的人工智能技能了。
需要注意的是，ML.Net是符合.Net Standard2.0标准的，如果在.Net Framework中使用，需要注意版本>=4.6.1

本文只代表作者本人理解，如有出入，欢迎在评论区指出，不拉不踩，交流为主

> 本文中出现的代码已经上传至github ： [https://github.com/BigHeadDev/ML.Net.Demo](https://github.com/BigHeadDev/ML.Net.Demo)
