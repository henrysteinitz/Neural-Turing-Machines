# Neural Turing Machines

## Overview

A Neural Turing Machine is neural network with augmented memory invented by
Alex Graves, Greg Wayne, and Ivo Danihelka at DeepMind in 2014. It can be
summarized as a Turing Machine in which the finite-state controller is replaced
with a simple neural network and the read/write operations are made
differentiable. These adjustments allow the system to compute error
derivatives using backpropagation so it can learn by gradient descent.

## How It Works
