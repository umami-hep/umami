In this part, we will focus on the training process, which involves preparing and running the model training using Keras.

Before starting the actual training, it's essential to set up our environment correctly. 
To prepare for training, run the following command in your terminal.

```bash
train.py -c examples/tutorial_jetclass/Train-config.yaml --prepare
```

This command will not directly start the training, but will prepare the model folder with all needed configs/scale dicts.

After preparing your training environment, it's time to start training your model.

Execute the following command to begin the training process. This command reads the same configuration file and starts training the model based on the parameters defined.

```bash
train.py -c examples/tutorial_jetclass/Train-config.yaml
```

As the training progresses, you will see output in the terminal indicating the training and validation loss, accuracy, and other metrics defined in your configuration file. This output helps monitor the model's performance and make any necessary adjustments.

Once the training is complete, the next step in the machine learning pipeline is to validate the model's performance. In the validation section, which follows this tutorial, you'll learn how to evaluate your trained model against a test dataset to understand its effectiveness and make any necessary improvements.
