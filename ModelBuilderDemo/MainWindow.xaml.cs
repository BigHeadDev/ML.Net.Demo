using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace ModelBuilderDemo {
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window {
        public MainWindow() {
            InitializeComponent();
        }

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
    }
}
