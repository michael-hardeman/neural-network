--  Copyright (C) 2025 Michael Hardeman
--
--  This library is free software; you can redistribute it and/or
--  modify it under the terms of the GNU Library General Public
--  License as published by the Free Software Foundation; either
--  version 2 of the License, or (at your option) any later version.
--
--  This library is distributed in the hope that it will be useful,
--  but WITHOUT ANY WARRANTY; without even the implied warranty of
--  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
--  Library General Public License for more details.
--
--  You should have received a copy of the GNU Library General Public
--  License along with this library; if not, write to the
--  Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
--  Boston, MA  02110-1301, USA.

with Ada.Numerics.Elementary_Functions; use Ada.Numerics.Elementary_Functions;

package Neural_Net is

   type Math_Func is not null access function (X : Float) return Float;

   --  Linear
   --  f(x) = alpha * x
   --  A simple linear transformation that scales the input by a constant factor.
   --  Often used in regression output layers or as identity functions when Alpha = 1.0.
   generic
      Alpha : Float;
   function Linear (X : Float) return Float;

   function Half_X (X : Float) return Float;     --  f(x) = 0.5 * x
   function One_X (X : Float) return Float;      --  f(x) = x (identity function)
   function Two_X (X : Float) return Float;      --  f(x) = 2.0 * x
   function Negative_X (X : Float) return Float; --  f(x) = -x (sign flip)

   --  E to Power
   --  f(x) = e^x
   --  Natural exponential function, used in various activation functions.
   function E_To_Power (X : Float) return Float is (2.71828 ** X);

   --  Sigmoid
   --  f(x) = 1 / (1 + e^-x)
   --  compresses input values into a range between 0 and 1.
   --  This function is often employed in the output layer of binary classification problems,
   --  as it yields outputs that resemble probabilities.
   function Sigmoid (X : Float) return Float is (1.0 / (1.0 + E_To_Power (-X)));

   --  Hyperbolic Tangent (Tanh)
   --  f(x) = (e^x - e^-x) / (e^x + e^-x)
   --  maps input values to the range [-1, 1]. It is frequently used within hidden layers of
   --  neural networks and can help alleviate the vanishing gradient problem when compared to sigmoid.
   function Hyperbolic_Trangent (X : Float) return Float is (
      declare Pos_Pow : constant Float := E_To_Power (X);
              Neg_Pow : constant Float := E_To_Power (-X);
      begin (Pos_Pow - Neg_Pow) / (Pos_Pow + Neg_Pow));

   --  Rectified Linear Unit (ReLU)
   --  f(x) = max(0, x),
   --  ReLU stands as one of the most popular activation functions.
   --  It introduces sparsity by setting negative values to zero, making it computationally
   --  efficient and well-suited for deep networks. However, it is not without its shortcomings,
   --  such as the dying ReLU problem.
   function Rectified_Linear_Unit (X : Float) return Float is (Float'Max (0.0, X));

   --  Leaky Rectified Linear Unit (Leaky ReLU)
   --  f(x) = max(alpha * x, x) where alpha is a small positive constant
   --  Leaky ReLU, an enhancement over standard ReLU, allows a small gradient for negative values.
   --  It is a useful choice when confronted with the dying ReLU problem.
   generic
      Alpha : Float;
   function Leaky_Rectified_Linear_Unit (X : Float) return Float;

   --  Exponential Linear Unit (ELU),
   --  f(x) = x if x > 0 else alpha * (e^x - 1),
   --  combines the strengths of ReLU and Leaky ReLU while mitigating the dying ReLU problem.
   --  It particularly shines when the network needs to capture both positive and negative values.
   generic
      Alpha : Float;
   function Exponential_Linear_Unit (X : Float) return Float;

   --  Swish
   --  f(x) = x * sigmoid(x)
   --  Proposed by Google's research team, it combines the advantages of ReLU's computational efficiency with a
   --  smoother, non-monotonic behavior, potentially leading to improved performance.
   function Swish (X : Float) return Float is (X * Sigmoid (X));

   --  Swish-1
   --  f(x) = x / (1 + exp(-x))
   --  A variant of swish. It introduces a division operation, which can provide different
   --  properties compared to standard Swish.
   function Swish_Minus_One (X : Float) return Float is (X / (1.0 + E_To_Power (-X)));

   --  Inverse Square Root Linear Unit (ISRLU)
   --  f(x) = x / sqrt(1 + x^2)
   --  It is another smooth alternative to ReLU.
   function Inverse_Square_Root_Linear_Unit (X : Float) return Float is (X / Sqrt (1.0 + (X ** 2)));

   --  Sigmoid derivative function
   --  f'(x) = sigmoid(x) * (1 - sigmoid(x))
   --  Computes the derivative of the sigmoid function, required for backpropagation.
   --  Used to calculate gradients during the backward pass.
   function Sigmoid_Derivative (X : Float) return Float is
      (declare S : constant Float := Sigmoid (X); begin S * (1.0 - S));

   --  ReLU derivative function
   --  f'(x) = 1 if x > 0 else 0
   --  Computes the derivative of the ReLU function, required for backpropagation.
   --  The derivative is 1 for positive inputs and 0 for negative inputs.
   function ReLU_Derivative (X : Float) return Float is (if X > 0.0 then 1.0 else 0.0);

   type Float_Array is array (Positive range <>) of Float;
   type Float_Matrix is array (Positive range <>, Positive range <>) of Float;

   --  Core data structure representing a single layer in the neural network
   type Layer_State (Layer_Size : Positive; Previous_Layer_Size : Positive) is record
      --  Forward propagation data
      Weights          : Float_Matrix  (1 .. Layer_Size, 1 .. Previous_Layer_Size); --  Connection weights from previous layer
      Biases           : Float_Array   (1 .. Layer_Size);                           --  Bias values for each neuron in this layer
      Input_Sums       : Float_Array   (1 .. Layer_Size);                           --  Weighted sum of inputs (before activation)
      Outputs          : Float_Array   (1 .. Layer_Size);                           --  Final neuron outputs (after activation)

      --  Backpropagation data (gradients for training)
      Weight_Gradients : Float_Matrix  (1 .. Layer_Size, 1 .. Previous_Layer_Size); --  Gradients for weight updates
      Bias_Gradients   : Float_Array   (1 .. Layer_Size);                           --  Gradients for bias updates
      Error_Deltas     : Float_Array   (1 .. Layer_Size);                           --  Error terms for backpropagation

      --  Function pointers for computations
      Activation       : Math_Func; --  Activation function applied to weighted sums
      Derivative       : Math_Func; --  Derivative of activation function for backpropagation
   end record;

   type Layer_Access is not null access all Layer_State;

   --  Because Layer_State is parameterized I cannot use it here unless I provide the parameters.
   --  I cannot provide default parameters to Layer_State because that causes storage errors if
   --  you want to allocate more than the default parameter. For now I will use Access types
   --  until I learn how to deal with this situation better.
   type Layer_Array is array (Positive range <>) of Layer_Access;

   --  Complete neural network structure with input storage
   --  This design stores the input values within the network, enabling proper
   --  gradient computation for the first layer during backpropagation.
   type Neural_Network (Input_Count : Positive; Layer_Count : Positive) is record
      Input  : Float_Array (1 .. Input_Count);   --  Stored input values for gradient computation
      Layers : Layer_Array (1 .. Layer_Count);   --  Array of computational layers
   end record;

   --  Performs forward propagation through the entire network
   --  Computes outputs for each layer sequentially from input to output.
   --  Can be used to run the network and take inputs to produce an output.
   procedure Forward_Pass (Network : in out Neural_Network; Input : Float_Array);

   --  Performs backward propagation (backpropagation) to update network weights
   --  Computes gradients and updates weights/biases based on the difference between
   --  actual output and target values.
   procedure Backward_Pass (Network : in out Neural_Network;
                            Target : Float_Array;
                            Learning_Rate : Float);

   --  Performs one complete training step (forward pass + backward pass)
   --  Convenience function that combines forward propagation and backpropagation
   --  for a single training example.
   procedure Train_Step (Network : in out Neural_Network;
                         Input : Float_Array;
                         Target : Float_Array;
                         Learning_Rate : Float);

   --  Retrieves the final output values from the network's output layer
   function Get_Network_Output (Network : Neural_Network) return Float_Array;

   --  Calculates Mean Squared Error (MSE) loss between network output and target values
   --  MSE = average((output - target)^2)
   function Calculate_Loss (Output, Target : Float_Array) return Float;

end Neural_Net;