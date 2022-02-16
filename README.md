## Study on the Impact of Random Labelling in Detecting Image Distortions

With a purpose to mimic visual distortions on medical images, we have conducted a study to derive possible relationships between random labelling ratio and model accuracy.

### Background

Since the discovery of X-Ray in the 1890s, medical imaging techniques have been widely used in the medical community, aiming to help doctors and clinicians provide diagnosis and further medical treatments for patients. In general, these medical scans assist doctors in both visual cognition and perception.

Unfortunately, visual distortion effects can sometimes happen to these medical images, making it hard for doctors to provide accurate presciptions for patients and possibly causing severe outcomes. In order to prevent this, it is rather important for us to detect those distorted areas in these radiographs and derive the relationship between random false labelling ratio and diagnosis accuracy.

However, there is no ready-made vision datasets meant for detecting image distortions. Therefore, we have to generate these medical scans with distortions by ourselves. Unable to access the original medical images, we decide to mimic the visual distortions on the classic PASCAL VOC 2007 and 2012 datasets. However, in order to keep our original train set completely clean, we prefer an artificial data generator to labels and annotations provided by human beings as the it is inevitable for humans to make annotation errors. For the distortion classes, we decide to go with 4 different classes, i.e., Blob, Blur, Channel Change and Distortion. For the Blob class, differently sized blobs restricted in a random polygon are injected into the original pixels. For images labelled with Blur, we apply the GaussianBlur function on a random selected polygons within the image dimension. For the 
