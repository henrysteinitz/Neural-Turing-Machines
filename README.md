# Neural Turing Machines

## Overview

A Neural Turing Machine (NTM) is neural network with augmented memory invented by
Alex Graves, Greg Wayne, and Ivo Danihelka at DeepMind in 2014. It's essentially
a Turing Machine whose finite-state controller is replaced
with a simple feedforward or recurrent neural network. Differentiable versions
of the read/write operations are then attached to the new controller, allowing
the NTM to compute error derivatives using backpropagation. This means the
entire system can learn by gradient methods.

## How It Works
