## Convolutional Neural Network for Graviational Wave Analysis

### Abstract:

For my final project in EECS 349 (Northwestern University), I investigated the use of neural networks, specifically convolutional neural networks, in gravitational wave data analysis related to the Laser Interferometer Space Antenna (LISA). I worked on this project by myself. My contact email address is mikekatz04@gmail.com. 

LISA is a future space based mission to measure gravitational waves travelling from the most extreme objects in the universe: black holes, neutron stars, and white dwarf stars. While the mission is underway, the detector will be detecting, in a singular datastream, upwards of 10000+ gravitational wave sources simultaneously. One of the main sources of interest are Extreme Mass Ratio Inspirals (EMRIs). EMRIs occur when a small black hole, like those resulting from dying stars, orbit at close to the speed of light around massive black holes existing in the centers of galaxies. The mass ratio between the binaries is usually 10000 or more. These orbits are very complex, and are still not well understood theoretically or computationally. These sources will help shed light on extreme dynamical systems in the centers of galaxies, as well as help refine our understanding of the General Theory of Relativity. The state of data analysis for EMRIs is wide open. I wanted to begin to investigate using machine learning to help us accomplish many sorts of difficult data related tasks. 

I focused on the amplitude of the fourier transform of an EMRI signal for the features in this project. Two tasks were of most interest to me in this project. The first task was to use regression to predict the three fundamental frequencies that make up the complex time-domain EMRI signal. To do this, I used a 1-dimensional convolutional neural network with the amplitudes in each frequency bin as the features and the theoretically determined fundamental freqeuncies as the labels. I additionally used a nearest neighbors algorithm for reference. However, the goal was to make a scalable project that can query fast with the data after many input waveforms. A nearest neighbors regressor is not expected to scale well in terms of query time, and, therefore, was not the main focus of the project. 

The second aspect of the project was to use a convolutional autoencoder to reduce the dimensionality of the amplitude spectrum from 800 to lower dimensions. This is for the purpose of using future regression studies with techniques such as gaussian process regression. For this task, the features were the same as the previous task, but the labels were the features themselves, as in any autoencoder. Below are two images showing the output of the autoencoder compared to the initial amplitudes. 

![](images/autoencoder_img_5.png?raw=true)

The amplitude spectrum is shown above for a binary with low eccentricity (0.028) and a lower inclination (0.42 radians). The prediction from the autoencoder is shown in dotted orange. 


![](images/autoencoder_img_90.png?raw=true)

The amplitude spectrum is shown above for a binary with a higher eccentricity (0.37) and a lower inclination (0.35 radians). The prediction from the autoencoder is shown in dotted orange. Higher eccentricities lead to signals with higher order harmonics.


### Data Creation

For this project, I generated datasets using a waveform genearator by Chua et al 2017. I used the numerical kludge waveform, which is fast compared to high fidelity waveforms, but slow compared to analytical generators. I fixed all of the parameters of the waveform generator except for the eccentricity an inclination. The eccentricity defines the ellipticity of the orbit away from circularity. I sampled 12 hrs of waveform data on a grid of eccentricities from 0.01 to 0.4 and inclinations from 0.01 to 80 degrees. My training set consisted of 	10000 of these waveforms. I then used numpy's discrete fourier transform to find the amplitude spectrum of each waveform. The signals vary from 0.1 mHz to 100 mHz. Therefore, I cut down the dataset from 21601 frequency bins to the 859 bins within those two limiting frequencies. I then normalized the amplitude spectrum so the convolutional network could focus on the shapes of each waveform rather than the overall magnitude. This is because extrinsic quantities like the distance can modulate the signal amplitude; I wanted this inital trainer to focus on the intrinsic quanties specific to each system. 

In addition to the amplitude spectrum, I also had the generator output the trajectory of the orbit so that I knew the eccentricity and inclination. With these values in hand, I was able to find the theoretical fundamental frquencies of the orbit which are determined from orbital parameters. These values became the labels attached to each amplitude spectrum. At training time, I rescaled the ranges of the three fundamental frequency to zero to one. 

Due to the computation time for generating these waveforms, I decided to generate a smaller test set than I plan to use in the future. Within the eccentricity and inclination limits stated above, I drew each from a uniform distribution each parameter to generate 100 test waveforms. Once again, I found the fundamental frequencies of the orbit in the same way as mentioned above.


### Fundamental Frequency Regressor

#### Methods

One issue with fourier transforms is spectral leakage between neighboring bins. This makes finding peaks an issue. Additionally, the bins are not centered on the integer multiples of the fundamental frequencies (higher order harmonics). Therefore, locating peaks can only help determine integer multiples of frequencies to a degree of certainty equal to the bin width. Also, more peaks helps further determine the system of equations we could algebraically use to determine the frequencies. However, lower eccentricity signals radiate at a small amount of frequencies, making the system of equations impossible to solve. 

Therefore, machine learning can be used to learn the shapes of amplitude spectral peaks and relate them to actual fundamental frequency values, allowing for a regressor that can handle spectral leakage and a small amount of peaks. For this, it was clear I needed to use a convolutional neural network that could understand the shape of the 1D spectrum, could associate neighboring features to understand where peaks are and how to match them to signals from the test set. I tried many different architectures of the convolutional network to see which would be the best for this initial investigation. I did not use cross-validation due to computation time and the fact that my training set was gridded, which is similar to what would be done in reality. However, this is an interesting investigation to locate which areas of the parameter space are more sensitive to less waveforms (see Futrure Work).

For all models I tried, I used 2 or 4 convolutional layers with a corresponding max pooling layer for each convolutional layer. I also tried different kernel sizes at 10 or 50. For initial convlutional layer, I used 32 filters. After this, each convolutinal layer had 64 filters. After the final layer max pooling layer, the tensor was flattened and then fed into a dense layer with 1024 neurons with ReLu activation. After this dense layer, I had a dropout layer to remove 40 percent of the previous layer's neurons. The output layer had 3 neurons, one for each fundamental frequency, with a linear activation for regression. For one model, I used ReLu activation on the output layer since the frequencies are positive definite. The loss function I used was the mean squared error. Each model was trained for 200 epochs.

In addition to the convolutional neural network, I also used nearest neighbor, which I thought would work well on this small scale within the grid of input parameters. However, outside of these simplistic settings, I do not expect this method to remain as accurate and fast. I used a three nearest neighbors regressor, with weights based on the distance to the three neighbors. With the three frequencies rescaled to zero to one, I used a euclidean metric, and an L-2 distance measure.

#### Results

Overall, all five models tested ran very similarly, which is encouraging. It seems the underlying function is not smooth. There is reason for this. The test samples are arranged by increasing eccentricity, but have varying inclinations. These two properties do not affect the phi frequency, but will after the theta and r frequencies, therefore, making these curves oscillate. In the future, I would look into more drop out layers to try and smooth it out, pending a deeper analysis of the overfitting. The plot below shows the six models I tested, including the three nearest neighbors, compared to the actual fundamental frequencies. The solid lines are the actuals while the dotted lines are the predictions. The fundamental frequency in phi is smoother, and better reproduced by the models. The frequency in the radial direction is very chaotic. So are the model fits. The over fitting would be interested to analyze compared to the 

![](images/model_outputs.png?raw=true)
This plot shows the fit for each of the 6 models to each frequency. The frequencies in phi, theta, and r are shown in blue, green, and orange, repectively. The kernel size, convolutional layers, and output layer activation is indicated in the title of each plot with K, CL, and Act, respectively.

To compare these models further, I find the mean error and standard deviation on the error for each frequency from each model. Models 1 through 6 are in order: (K=50, CL=2, Act=Lin), (K=50, CL=4, Act=Lin), (K=10, CL=2, Act=Lin), (K=10, CL=4, Act=Lin), (K=50, CL=2, Act=Relu), and 3NN. The mean error, indicating the average difference from the true value, means the higher the mean, the worse the fit. The standard deviation on the error indicates the precision of the model. Smaller deviation means more precise to a predictive nature. The behavior between the frequencies are consistent, with the largest errors for the frequency in the radial oscillation.  

The best models turn out to be 2 and 4, i.e. (K=50, CL=4, Act=Lin) and (K=10, CL=4, Act=Lin). Both of these models have four convolutional layers. This is interesting, because they actually have less parameters to fit. The added convolutions must give a bit more information for hte prediction. The precision on both of these measurements is quite good. 

The worst model is model 1, (K=50, CL=2, Act=Lin). This has the least number of convolutional layers and the largest kernel. Lowering the kernel or adding layers improves the accuracy of the fit. Additionally, model 5 has the same number of layers and kernel size as the model 1, with a different activation. This indicates replacing the linear activation on the output layer with a ReLu activation makes the model ore precise. This makes sense. All of the outputs are greater than zero; ReLu activation reinforces this requirement.

All neural network models 2 to 5 all perform better than nearest neighbors. This is quite encouraging for scaling. This means the faster query operation can perform better too. 


![](images/means_and_deviations.png?raw=true)
The mean errors and standard deviation on the errors for each model 1 through 6 are shown for each frequency. 
 







### Future Work

-Compare fast AAK generation of larger data vs. NK generation. 
-2D versions
-gridding parameter space
-classification if signal is in noise
-gaussian processes
-faster predictor of mapping from parameters to frequencies, back and forth. 
-add more parameters

Inline-style: 
!(https://github.com/mikekatz04/cnn_project/autoencoder_img_0.png "Logo Title Text 1")

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/mikekatz04/cnn_project/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
