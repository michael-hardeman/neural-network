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

package body Neural_Net is

   ------------
   -- Linear --
   ------------
   function Linear (X : Float) return Float is (Alpha * X);

   function Half_X_Inner is new Linear (Alpha => 0.5);
   function One_X_Inner is new Linear (Alpha => 1.0);
   function Two_X_Inner is new Linear (Alpha => 2.0);
   function Negative_X_Inner is new Linear (Alpha => -1.0);

   function Half_X (X : Float) return Float is (Half_X_Inner (X));
   function One_X (X : Float) return Float is (One_X_Inner (X));
   function Two_X (X : Float) return Float is (Two_X_Inner (X));
   function Negative_X (X : Float) return Float is (Negative_X_Inner (X));

   ---------------------------------
   -- Leaky_Rectified_Linear_Unit --
   ---------------------------------
   function Leaky_Rectified_Linear_Unit (X : Float) return Float is (Float'Max (Alpha * X, X));

   -----------------------------
   -- Exponential_Linear_Unit --
   -----------------------------
   function Exponential_Linear_Unit (X : Float) return Float is (if X > 0.0 then X else Alpha * (E_To_Power (X) - 1.0));

   ------------------
   -- Forward_Pass --
   ------------------
   procedure Forward_Pass (Network : in out Neural_Network; Input : Float_Array) is
      procedure Compute_Layer (Layer : in out Layer_State; Previous_Output : Float_Array) is begin
         for Neuron_Index in Layer.Outputs'Range loop
            declare
               Weighted_Sum : Float renames Layer.Input_Sums (Neuron_Index);
            begin
               --  Start with the initial bias
               Weighted_Sum := Layer.Biases (Neuron_Index);

               --  Calculate weighted sum of inputs
               for Input_Index in Previous_Output'Range loop
                  Weighted_Sum := Weighted_Sum +
                     (Layer.Weights (Neuron_Index, Input_Index) * Previous_Output (Input_Index));
               end loop;

               --  Apply activation function
               Layer.Outputs (Neuron_Index) := Layer.Activation (Weighted_Sum);
            end;
         end loop;
      end Compute_Layer;

      Last_Index : Positive := Network'First;
   ---------------------
   begin -- Forward_Pass
   ---------------------
      --  Network must have at least 1 layer
      Compute_Layer (Network (Network'First).all, Input);

      for Layer_Index in Network'First + 1 .. Network'Last loop
         Compute_Layer (Network (Layer_Index).all, Network (Last_Index).Outputs);
         Last_Index := Layer_Index;
      end loop;
   end Forward_Pass;

   -------------------
   -- Backward_Pass --
   -------------------
   procedure Backward_Pass (Network       : in out Neural_Network;
                            Target        : Float_Array;
                            Learning_Rate : Float) is
      procedure Compute_Output_Layer_Deltas (Layer : in out Layer_State; Target : Float_Array) is begin
         --  For output layer: delta = (output - target) * activation_derivative(input_sum)
         for I in Layer.Error_Deltas'Range loop
            declare
               Output_Error : constant Float := Layer.Outputs (I) - Target (I);
               Activation_Grad : constant Float := Layer.Derivative (Layer.Input_Sums (I));
            begin
               Layer.Error_Deltas (I) := Output_Error * Activation_Grad;
            end;
         end loop;
      end Compute_Output_Layer_Deltas;

      procedure Compute_Hidden_Layer_Deltas (Current_Layer : in out Layer_State;
                                             Next_Layer : Layer_State) is
      begin
         --  For hidden layers: delta = sum(next_layer_weights * next_layer_deltas) * activation_derivative
         for I in Current_Layer.Error_Deltas'Range loop
            declare
               Error_Sum : Float := 0.0;
               Activation_Grad : constant Float :=
                  Current_Layer.Derivative (Current_Layer.Input_Sums (I));
            begin
               --  Sum weighted errors from next layer
               for J in Next_Layer.Error_Deltas'Range loop
                  Error_Sum := Error_Sum + (Next_Layer.Weights (J, I) * Next_Layer.Error_Deltas (J));
               end loop;

               Current_Layer.Error_Deltas (I) := Error_Sum * Activation_Grad;
            end;
         end loop;
      end Compute_Hidden_Layer_Deltas;

      procedure Compute_Gradients (Layer : in out Layer_State; Previous_Output : Float_Array) is
      begin
         --  Compute weight gradients: gradient = delta * previous_layer_output
         for I in Layer.Weight_Gradients'Range (1) loop
            for J in Layer.Weight_Gradients'Range (2) loop
               Layer.Weight_Gradients (I, J) := Layer.Error_Deltas (I) * Previous_Output (J);
            end loop;
         end loop;

         --  Compute bias gradients: gradient = delta
         for I in Layer.Bias_Gradients'Range loop
            Layer.Bias_Gradients (I) := Layer.Error_Deltas (I);
         end loop;
      end Compute_Gradients;

      procedure Update_Weights (Layer : in out Layer_State; Learning_Rate : Float) is
      begin
         --  Update weights: weight = weight - learning_rate * gradient
         for I in Layer.Weights'Range (1) loop
            for J in Layer.Weights'Range (2) loop
               Layer.Weights (I, J) := Layer.Weights (I, J) -
                  (Learning_Rate * Layer.Weight_Gradients (I, J));
            end loop;
         end loop;

         --  Update biases: bias = bias - learning_rate * gradient
         for I in Layer.Biases'Range loop
            Layer.Biases (I) := Layer.Biases (I) - (Learning_Rate * Layer.Bias_Gradients (I));
         end loop;
      end Update_Weights;
   ----------------------
   begin -- Backward_Pass
   ----------------------
      --  Step 1: Compute deltas starting from output layer
      Compute_Output_Layer_Deltas (Network (Network'Last).all, Target);

      --  Step 2: Propagate deltas backward through hidden layers
      for Layer_Index in reverse Network'First .. Network'Last - 1 loop
         Compute_Hidden_Layer_Deltas (Network (Layer_Index).all,
                                      Network (Layer_Index + 1).all);
      end loop;

      --  Step 3: Compute gradients and update weights
      --  First layer uses original input (would need to store this)
      --  For now, we'll handle layers that get input from previous layers
      for Layer_Index in Network'First + 1 .. Network'Last loop
         Compute_Gradients (Network (Layer_Index).all,
                            Network (Layer_Index - 1).Outputs);
         Update_Weights (Network (Layer_Index).all, Learning_Rate);
      end loop;

      --  Handle first layer separately (needs original input - this is simplified)
      --  In a complete implementation, you'd store the original input
      if Network'Length > 0 then
         Update_Weights (Network (Network'First).all, Learning_Rate);
      end if;
   end Backward_Pass;

   ----------------
   -- Train_Step --
   ----------------
   procedure Train_Step (Network : in out Neural_Network;
                         Input : Float_Array;
                         Target : Float_Array;
                         Learning_Rate : Float) is
   begin
      Forward_Pass (Network, Input);
      Backward_Pass (Network, Target, Learning_Rate);
   end Train_Step;

   --------------------
   -- Calculate_Loss --
   --------------------
   function Calculate_Loss (Output, Target : Float_Array) return Float is
      Loss : Float := 0.0;
   begin
      --  Mean Squared Error
      for I in Output'Range loop
         declare
            Error : constant Float := Output (I) - Target (I);
         begin
            Loss := Loss + (Error * Error);
         end;
      end loop;
      return Loss / Float (Output'Length);
   end Calculate_Loss;

   ------------------------
   -- Get_Network_Output --
   ------------------------
   function Get_Network_Output (Network : Neural_Network) return Float_Array is begin
      return Network (Network'Last).all.Outputs;
   end Get_Network_Output;

end Neural_Net;