## Convolutional Neural Network for Graviational Wave Analysis

### Abstract:

For my final project in EECS 349 (Northwestern University), I investigated the use of neural networks, specifically convolutional neural networks, in gravitational wave data analysis related to the Laser Interferometer Space Antenna (LISA). I worked on this project by myself. My contact email address is mikekatz04@gmail.com. 

LISA is a future space based mission to measure gravitational waves travelling from the most extreme objects in the universe: black holes, neutron stars, and white dwarf stars. While the mission is underway, the detector will be detecting, in a singular datastream, upwards of 10000+ gravitational wave sources simultaneously. One of the main sources of interest are Extreme Mass Ratio Inspirals (EMRIs). EMRIs occur when a small black hole, like those resulting from dying stars, orbit at close to the speed of light around massive black holes existing in the centers of galaxies. The mass ratio between the binaries is usually 10000 or more. These orbits are very complex, and are still not well understood theoretically or computationally. These sources will help shed light on extreme dynamical systems in the centers of galaxies, as well as help refine our understanding of the General Theory of Relativity. The state of data analysis for EMRIs is wide open. I wanted to begin to investigate using machine learning to help us accomplish many sorts of difficult data related tasks. 

I focused on the amplitude of the fourier transform of an EMRI signal for the features in this project. Two tasks were of most interest to me in this project. The first task was to use regression to predict the three fundamental frequencies that make up the complex time-domain EMRI signal. To do this, I used a 1-dimensional convolutional neural network with the amplitudes in each frequency bin as the features and the theoretically determined fundamental freqeuncies as the labels. I additionally used a nearest neighbors algorithm for reference. However, the goal was to make a scalable project that can query fast with the data after many input waveforms. A nearest neighbors regressor is not expected to scale well in terms of query time, and, therefore, was not the main focus of the project. 

The second aspect of the project was to use a convolutional autoencoder to reduce the dimensionality of the amplitude spectrum from 800 to lower dimensions. This is for the purpose of using future regression studies with techniques such as gaussian process regression. For this task, the features were the same as the previous task, but the labels were the features themselves, as in any autoencoder. Below are two images showing the output of the autoencoder compared to the initial amplitudes. 

![](images/autoencoder_img_5.png?raw=true)	![](images/autoencoder_img_90.png?raw=true)

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
