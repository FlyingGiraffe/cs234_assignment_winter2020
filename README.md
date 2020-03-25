# CS234 Assignment
Assignments for CS234 Reinforcement Learning

## Completed
+ **Assignment 1 - Frozen Lake**
+ **Assignment 2 - Atari** <br>
  Something tricky in `q3_nature.py`: `tf.contrib.layers.conv2d` works much better than `tf.layers.conv2d` here, though I don't know why.
  Some differences between these two functions are:
  + **performance:** after training the model for 5 million steps, we get a total return of around +15 if we use `tf.contrib.layers.conv2d`, but around +0 if we use `tf.layers.conv2d`;
  + `tf.contrib.layers.conv2d` uses `padding='SAME'` by default, while `tf.layers.conv2d` uses `padding='VALID'` by default;
  + saving the weights for the model, we get a file of around 50MB if `tf.contrib.layers.conv2d` is used, but around 10MB if `tf.layers.conv2d` is used.
