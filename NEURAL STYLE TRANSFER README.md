#NEURAL STYLE TRANSFER

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: Swaraj Jha

*INTERN ID*: CTIS7922

*DOMAIN*: Python Programming

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

#Description: The neural style transfer implementation constituted a comprehensive computer vision pipeline engineered to synthesize visually compelling artistic renditions of photographs by algorithmically blending the semantic content structure of one image with the textural and stylistic characteristics of another through the optimization of deep convolutional neural network feature representations, specifically leveraging a pre-trained VGG19 architecture from the PyTorch torchvision ecosystem whose convolutional layers had been previously trained on the ImageNet dataset for object classification and thus possessed rich hierarchical feature extractors capable of disentangling high-level content semantics from low-level style textures. The system architecture was meticulously designed around an object-oriented paradigm encapsulating a Normalization module that standardized input tensors using ImageNet mean and standard deviation vectors, custom ContentLoss and StyleLoss classes that computed mean squared error discrepancies between target and generated feature maps—with style representation quantified through Gram matrices capturing channel-wise correlation statistics across spatial dimensions—and a model construction function that surgically inserted these loss modules at specific convolutional layers within a deep copy of the VGG19 feature extractor, namely conv_4 for content preservation and conv_1 through conv_5 for multi-scale style aggregation. The optimization strategy employed the Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm via PyTorch's LBFGS optimizer operating directly on the pixel values of an input image initialized as a clone of the content photograph, iteratively adjusting RGB intensities over three hundred steps to minimize a composite loss function weighted by a style coefficient of one million and a content coefficient of one, thereby balancing artistic abstraction against photographic fidelity while clamping tensor values to valid ranges after each iteration. Practical implementation considerations included PIL-based image ingestion with aspect-ratio-preserving resizing to five hundred twelve pixels, tensor transformations via torchvision composing resize, toTensor, and normalize operations, bidirectional conversion utilities between PIL Images and normalized PyTorch tensors, automatic style image interpolation to match content dimensions, and file system operations saving the final stylized output as a high-quality JPEG in the content image's parent directory. The script featured a terminal-based user interface prompting for absolute file paths, comprehensive error handling for missing files, and real-time console feedback displaying optimization progress with style and content loss metrics every fifty iterations, while remaining hardware-agnostic through automatic CUDA detection with CPU fallback, ultimately delivering a self-contained Python script requiring only torch, torchvision, and Pillow installations to transform arbitrary photographs into works mimicking the aesthetic qualities of user-provided artwork.

#Output: 
