## Study on the Impact of Random Labelling in Detecting Image Distortions

With a purpose to mimic visual distortions on medical images, we have conducted a study to derive possible relationships between random labelling ratio and model accuracy.

### Background

Since the discovery of X-Ray in the 1890s, medical imaging techniques have been widely used in the medical community, aiming to help doctors and clinicians provide diagnosis and further medical treatments for patients. In general, these medical scans assist doctors in both visual cognition and perception.

Unfortunately, visual distortion effects can sometimes happen to these medical images, making it hard for doctors to provide accurate presciptions for patients and possibly causing severe outcomes. In order to prevent this, it is rather important for us to detect those distorted areas in these radiographs and derive the relationship between random false labelling ratio and diagnosis accuracy.

However, there is no ready-made vision datasets meant for detecting image distortions. Therefore, we have to generate these medical scans with distortions by ourselves. Unable to access the original medical images, we decide to mimic the visual distortions on the classic PASCAL VOC 2007 and 2012 datasets. However, in order to keep our original train set completely clean, we prefer an artificial data generator to labels and annotations provided by human beings as the it is inevitable for humans to make annotation errors. For the distortion classes, we decide to go with 4 different classes, i.e., Blob, Blur, Channel Change and Distortion. For the Blob class, differently sized blobs restricted in a random polygon are injected into the original pixels. For images labelled with Blur, we apply the GaussianBlur function on a random selected polygons within the image dimension. For the Channel Change class, we simple choose a shuffeled color channel and inject it to the randomly picked area. Last but not least, images labelled with Distortion are "polluted" by randomly picking a color channel and adding/subtracting a product constituted by a distortion parameter and 1/-1.

### Models to use
In my case, all the empirical experiments are done with the SSD300 model due to limited training resources.

Do not know whether details of SSD model should be introduced, so I just leave it as blank.

### Application study

When it comes to the evalution metric, we use the mean Average Precision (mAP) as it is one of the most popular evaluation criterion used in the context of object recognition.

The APs acquired for the 4 image classes can be found in the following figure.

![APs](./evaluate/AP_Layout.png "APs obtained for the 4 distortion classes")

As can be clearly seen from the above graph, our model is better at detecting images labelled with Blur and Blob. Regarding the resistance study on how sensitive differently distorted images are to random labelling in our empirical study. The result can be displayed with a boxplot (here "cc" denotes the class of Channel Change):
![boxplot](./evaluate/boxplot_update1.png "APs obtained for the 4 distortion classes")
In our study, especially images labelled with class Blur are more resistant to random labelling in general. Conversely, images labelled with Channel Change show more variation in the model accuracy and therefore are more sensitive to randomly distributed noise.

Since model accuracy is measured with mAP, we are also giving performance curves based on it.
![mAP](./evaluate/mAP_Analysis_new.png "APs obtained for the 4 distortion classes")

Seen from the above curves, mAPs are taking on a fluctuated trend with an ascending amount of randomly distributed noise injected in the original train dataset. In this case, it is hard for us to derive a rigid relationship between random labelling ratio and model accuracy. Nevertheless, our neural network, SSD300, has shown resistance to random label noise as only a variation of around 1% in accuracy is found in the sets of emperiments.

