# Neural Networks with PyTorch - Complete Learning Repository

A comprehensive collection of Jupyter notebooks for learning Neural Networks using PyTorch, from beginner to advanced levels.

## 📚 Overview

This repository contains three progressive notebooks designed to take you from understanding the basics to implementing practical neural network solutions:

1. **`neural_networks_simple.ipynb`** - Beginner-friendly introduction
2. **`neural_networks_practical.ipynb`** - Real-world fraud detection application
3. **`neural_networks_advanced.ipynb`** - Advanced concepts and techniques

> **Note**: This repository focuses on hands-on implementation. For conceptual explanations, terminologies, and PyTorch fundamentals, please refer to the accompanying PowerPoint presentation.

---

## 📖 Notebook Descriptions

### 1. Neural Networks Made Simple (`neural_networks_simple.ipynb`)

**Target Audience**: Beginners and intermediate learners  
**Duration**: ~2-3 hours  
**Difficulty**: ⭐⭐ (Easy to Medium)

#### What You'll Learn:
- **Perceptron to Neural Network**: Understanding the building blocks
- **Activation Functions**: ReLU, Tanh, Sigmoid with visualizations
- **Training Loop**: Step-by-step implementation
- **Optimizers**: SGD vs Adam comparison
- **Regularization**: Dropout explained simply
- **Visualizations**: Decision boundaries, loss curves, and more

#### Key Features:
- Simple, easy-to-understand code
- Extensive visualizations
- Clear markdown explanations
- Minimal complexity
- Perfect for classroom demos

#### Topics Covered:
- Perceptron implementation
- XOR problem demonstration
- Activation functions (ReLU, Tanh, Sigmoid)
- Building your first neural network
- Complete training loop
- Optimizer comparison (SGD vs Adam)
- Dropout for overfitting prevention
- Moons dataset classification

---

### 2. Neural Networks Practical (`neural_networks_practical.ipynb`)

**Target Audience**: Intermediate to advanced learners  
**Duration**: ~2-3 hours  
**Difficulty**: ⭐⭐⭐ (Medium)

#### What You'll Learn:
- **Real-World Application**: Fraud detection problem
- **PyTorch Built-ins**: Using `nn.Sequential`, `nn.Linear`, etc.
- **Feed Forward Propagation**: Automatic forward pass
- **Backpropagation**: Automatic gradient computation
- **Overfitting/Underfitting**: Detection and prevention
- **Model Evaluation**: Comprehensive metrics
- **Parameter Tuning**: Learning rate and architecture optimization

#### Key Features:
- Uses only PyTorch's built-in functions (no custom classes)
- Real-world fraud detection dataset
- Minimal, precise, and well-commented code
- Comprehensive evaluation metrics
- Parameter tuning demonstrations

#### Topics Covered:
- Data preprocessing and standardization
- Building neural networks with `nn.Sequential`
- Feed forward propagation demonstration
- Training loop with automatic backpropagation
- Training progress visualization
- Model evaluation (Accuracy, Precision, Recall, F1-Score)
- Confusion matrix analysis
- Dropout for overfitting prevention
- Learning rate tuning
- Architecture comparison (Small, Medium, Large)

#### Dataset:
- **Type**: Synthetic credit card fraud detection
- **Samples**: 10,000 transactions
- **Features**: 30 transaction characteristics
- **Classes**: Normal (99%) vs Fraud (1%) - Imbalanced dataset

---

### 3. Neural Networks Advanced (`neural_networks_advanced.ipynb`)

**Target Audience**: Advanced learners  
**Duration**: ~3-4 hours  
**Difficulty**: ⭐⭐⭐⭐ (Advanced)

#### What You'll Learn:
- **Advanced Architectures**: Deep networks, residual connections
- **Advanced Optimization**: Learning rate scheduling, weight initialization
- **Advanced Regularization**: Batch normalization, advanced dropout techniques
- **Debugging Techniques**: Gradient inspection, activation analysis
- **Performance Optimization**: GPU utilization, batch processing
- **Advanced Evaluation**: ROC curves, precision-recall curves

#### Key Features:
- Deep network architectures
- Advanced PyTorch features
- Performance optimization techniques
- Comprehensive debugging tools
- Production-ready practices

#### Topics Covered:
- Deep neural network architectures
- Batch normalization
- Advanced dropout techniques
- Learning rate scheduling
- Weight initialization strategies
- Gradient clipping
- Early stopping
- Model checkpointing
- Advanced evaluation metrics
- Performance optimization

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Basic understanding of Python programming
- Basic understanding of machine learning concepts (for practical and advanced notebooks)

### Installation

1. **Clone or download this repository**

2. **Install required packages**:
   ```bash
   pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn jupyter
   ```

   Or install packages directly in the notebooks (each notebook includes installation cells)

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. **Start with the appropriate notebook**:
   - **Beginner**: Start with `neural_networks_simple.ipynb`
   - **Intermediate**: Start with `neural_networks_practical.ipynb`
   - **Advanced**: Start with `neural_networks_advanced.ipynb`

---

## 📋 Learning Path

### Recommended Sequence:

1. **Start Here**: `neural_networks_simple.ipynb`
   - Build foundational understanding
   - Learn basic concepts with simple examples
   - Get comfortable with PyTorch basics

2. **Apply Knowledge**: `neural_networks_practical.ipynb`
   - Apply concepts to a real-world problem
   - Learn to use PyTorch built-in functions
   - Understand evaluation and tuning

3. **Master Advanced Topics**: `neural_networks_advanced.ipynb`
   - Deep dive into advanced techniques
   - Learn production-ready practices
   - Master optimization and debugging

### Alternative Paths:

- **Quick Start (Practical Focus)**: Jump directly to `neural_networks_practical.ipynb` if you have basic ML knowledge
- **Theory First**: Review the PowerPoint presentation before starting any notebook
- **Reference Mode**: Use notebooks as reference guides for specific topics

---

## 🎯 Key Concepts Covered Across All Notebooks

### Core Concepts:
- ✅ Feed Forward Propagation
- ✅ Backpropagation (Automatic in PyTorch)
- ✅ Loss Functions (CrossEntropyLoss, MSELoss)
- ✅ Optimizers (SGD, Adam, RMSProp)
- ✅ Activation Functions (ReLU, Tanh, Sigmoid)
- ✅ Regularization (Dropout, Batch Normalization)
- ✅ Model Evaluation (Metrics, Confusion Matrix)
- ✅ Parameter Tuning (Learning Rate, Architecture)

### PyTorch Features:
- ✅ `nn.Module` and `nn.Sequential`
- ✅ `nn.Linear`, `nn.ReLU`, `nn.Dropout`
- ✅ `torch.optim` optimizers
- ✅ Automatic differentiation (`loss.backward()`)
- ✅ Training vs Evaluation modes (`model.train()`, `model.eval()`)
- ✅ GPU support (in advanced notebook)

---

## 📊 Notebook Comparison

| Feature | Simple | Practical | Advanced |
|---------|--------|-----------|----------|
| **Target Level** | Beginner | Intermediate | Advanced |
| **Code Complexity** | Low | Medium | High |
| **Real-World Dataset** | ❌ | ✅ | ✅ |
| **Custom Classes** | ✅ | ❌ | ✅ |
| **Built-in Functions** | ✅ | ✅ | ✅ |
| **Visualizations** | Extensive | Moderate | Advanced |
| **Parameter Tuning** | Basic | Comprehensive | Advanced |
| **Evaluation Metrics** | Basic | Comprehensive | Advanced |
| **Debugging Tools** | Basic | Basic | Advanced |

---

## 🛠️ Usage Instructions

### For Each Notebook:

1. **Read the Introduction**: Understand the learning objectives
2. **Run Setup Cells**: Install packages and import libraries
3. **Follow Sequentially**: Execute cells in order
4. **Read Comments**: Code is well-commented for learning
5. **Experiment**: Modify parameters and observe results
6. **Review Visualizations**: Understand what the plots show

### Tips for Learning:

- **Don't Skip Cells**: Each cell builds on previous ones
- **Read Markdown**: Explanations are in markdown cells
- **Experiment**: Change hyperparameters and see what happens
- **Take Notes**: Document your learnings
- **Review Code**: Understand what each line does

---

## 📁 Repository Structure

```
ML_Artifacts/
│
├── README.md                          # This file
├── neural_networks_simple.ipynb       # Beginner notebook
├── neural_networks_practical.ipynb    # Practical notebook
├── neural_networks_advanced.ipynb      # Advanced notebook
├── Neural_networks_pytorch.pptx     # Conceptual explanations (to be added)

```

---

## 🔧 Troubleshooting

### Common Issues:

1. **Import Errors**:
   - Solution: Run the installation cell in each notebook
   - Ensure you're using Python 3.7+

2. **CUDA/GPU Errors**:
   - Solution: Notebooks work on CPU by default
   - GPU is optional (only in advanced notebook)

3. **Memory Issues**:
   - Solution: Reduce batch size or number of samples
   - Close other applications

4. **Different Results**:
   - Solution: Set random seeds (already included in notebooks)
   - Results should be reproducible

---

## 📚 Additional Resources

### Recommended Reading:
- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

### For Conceptual Understanding:
- Refer to the accompanying PowerPoint presentation
- Covers: Neural network concepts, terminologies, PyTorch fundamentals

---

## 🎓 Learning Objectives

By completing all three notebooks, you will be able to:

1. ✅ Understand how neural networks work
2. ✅ Build neural networks using PyTorch
3. ✅ Implement complete training loops
4. ✅ Handle real-world datasets
5. ✅ Evaluate model performance
6. ✅ Tune hyperparameters effectively
7. ✅ Prevent overfitting
8. ✅ Debug neural networks
9. ✅ Optimize model performance
10. ✅ Apply neural networks to practical problems

---

## 🤝 Contributing

This is an educational repository created by author Varun Raste (@varunchach). Suggestions and improvements are welcome!

### How to Contribute:
- Report bugs or issues
- Suggest improvements
- Add more examples
- Improve documentation

---

## 📝 License

This repository is for educational purposes. Feel free to use and modify for learning.

---

## 👨‍🏫 For Instructors

### Teaching Recommendations:

1. **Start with PowerPoint**: Cover concepts and terminologies first
2. **Simple Notebook**: Use for in-class demonstrations
3. **Practical Notebook**: Assign as homework or lab work
4. **Advanced Notebook**: For advanced courses or self-study

### Assessment Ideas:
- Modify hyperparameters and observe changes
- Implement new architectures
- Apply to different datasets
- Compare different optimizers
- Analyze model performance

---

## 📞 Support

For questions or issues:
- Review the code comments (extensive documentation included)
- Check the PowerPoint presentation for conceptual clarity
- Refer to PyTorch documentation
- Experiment and learn by doing!

---

## 🎉 Happy Learning!

Start with `neural_networks_simple.ipynb` and work your way up. Remember: practice makes perfect!

**Good luck on your neural network journey! 🚀**

---

*Last Updated: 2025*  
*PyTorch Version: 2.8.0+*

