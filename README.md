# Neural Turing Machines

## Overview

A Neural Turing Machine is neural network with augmented memory invented by
Alex Graves, Greg Wayne, and Ivo Danihelka at DeepMind in 2014. It's essentially
a Turing Machine whose finite-state controller is replaced
with a simple neural network, and the read/write operations are made
differentiable. These adjustments allow the system to compute error
derivatives using backpropagation and thus learn by gradient descent.

## How It Works
