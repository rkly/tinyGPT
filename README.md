# 🤖 TinyGPT: Minimal GPT Implementation with TensorFlow

This playground project provides a simplified and minimalistic implementation of the Generative Pre-trained Transformer (GPT) using TensorFlow. The core components of the implementation are organized into the following files:

## 📁 Files

1. **gpt.py**: This file contains the implementation of the GPT model. It encapsulates the architecture and functionality of the Generative Pre-trained Transformer.

2. **train.py**: The `train.py` file houses the trainer object code. It orchestrates the training process for the GPT model by defining training parameters and executing the training loop.

3. **main.py**: Serving as the entry point to the application, `main.py` initializes the dataset and kicks off the training process. It brings together the essential components, making it easy to understand the workflow from dataset setup to training initiation.

4. **sample.py**: The `sample.py` script is designed for inference. After the model is trained, you can use this script to generate text based on a given prompt. It showcases the model's capabilities in a real-world scenario.

## 🛠 Dependencies

- **tensorflow** (tested on 2.15-post1): Core library for machine learning and deep neural networks.
- **tqdm**: Used for displaying a progress bar during training.
- **tensorflow_datasets**: Provides an example training dataset for quick experimentation (shakespeare).
- **absl**: Utilized for handling command-line flags.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The implementation is inspired by Andrej Karpathy's NanoGPT and its TensorFlow port by kamalkraj, as well as the TensorFlow documentation.

Happy coding! 🚀
