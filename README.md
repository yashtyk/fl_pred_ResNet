# ResNet
 Authors: Yana Shtyk, Olga Taran, André Csillaghy, Jonathan Donzallaz
 
 
The work addresses the problem of prediction of solar flares. To research this problem a machine learning technique called recurrent neural network was used.
We trained and tested our model on the SDOBenchmark dataset.  It is a time series dataset created by FHNW. 
It consists  of images of active regions cropped from SDO data. 
As the dataset has 10 different SDO channels, we were able to investigate prediction capabilities of all of them. Best TSS of 0.64 was obtained using a magnetogram. Furthermore, we found out that channel aggregation can help to improve prediction capabilities of the model. Using this technique we managed to achieve TSS of 0.7.

### How to run training process

#### Training on the one or multiple channels


##### Credit: Jonathan Donzallaz

Create cofiguration file \<filename\>.yaml with needed settings in the config/experiment folder and then run:
 ```

python main.py train +experiment=<filename>.yaml

```
 
### How to run testing process
 
#### Testing the model
 
##### Credit: Jonathan Donzallaz

 
 ```
 python main.py test +experiment=<filename>.yaml
 ```
 
 #### Testing performance of the models aggregation by summing models outputs
 
 
Create cofiguration file \<filename\>.yaml in the config/experiment folder with needed settings and paths to the models to be tested and then run:
 
 ```
 python main.py test_multi +experiment=<filename>.yaml
 ```
 
 #### Testing performance of the models aggregation by majority voting
 
 ### Reproducibility of the results
 Pretrained weights of the models can be downladed from [here](https://drive.google.com/drive/u/0/folders/1BVJRjiCydCIi-oLCZsBIOWrVNnzagmz2). They can be used to reproduce the results.
 
 
 
 
 
 
 
 
 
