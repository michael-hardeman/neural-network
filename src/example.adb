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

with Ada.Text_IO; use Ada.Text_IO;
with Neural_Net; use Neural_Net;

procedure Example is
   Input_Data : constant Float_Array := [0.5, 0.8];

   Hidden_Layer : aliased Layer_State := (
      Layer_Size => 3,
      Previous_Layer_Size => Input_Data'Length,
      Weights => [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
      Biases => [0.1, 0.2, 0.3],
      Input_Sums => [others => 0.0],
      Outputs => [others => 0.0],
      Weight_Gradients => [others => [others => 0.0]],
      Bias_Gradients => [others => 0.0],
      Error_Deltas => [others => 0.0],
      Activation => Sigmoid'Access,
      Derivative => Sigmoid_Derivative'Access);

   Output_Layer : aliased Layer_State := (
      Layer_Size => 1,
      Previous_Layer_Size => Hidden_Layer.Layer_Size,
      Weights => [[0.7, 0.8, 0.9]],
      Biases => [0.1],
      Input_Sums => [others => 0.0],
      Outputs => [others => 0.0],
      Weight_Gradients => [others => [others => 0.0]],
      Bias_Gradients => [others => 0.0],
      Error_Deltas => [others => 0.0],
      Activation => Sigmoid'Access,
      Derivative => Sigmoid_Derivative'Access);

   Network : Neural_Network := [Hidden_Layer'Unchecked_Access, Output_Layer'Unchecked_Access];
   Target_Data : constant Float_Array := [0.9];
   Learning_Rate : constant Float := 0.5;
   Epochs : constant := 100;
begin
   Put_Line ("Training Neural Network...");
   Put_Line ("Target Output: " & Target_Data (1)'Image);
   Put_Line ("");

   --  Show initial output
   Forward_Pass (Network, Input_Data);
   declare
      Initial_Output : constant Float_Array := Get_Network_Output (Network);
      Initial_Loss : constant Float := Calculate_Loss (Initial_Output, Target_Data);
   begin
      Put_Line ("Initial Output: " & Initial_Output (1)'Image);
      Put_Line ("Initial Loss: " & Initial_Loss'Image);
      Put_Line ("");
   end;

   --  Training loop
   for Epoch in 1 .. Epochs loop
      Train_Step (Network       => Network,
                  Input         => Input_Data,
                  Target        => Target_Data,
                  Learning_Rate => Learning_Rate);

      --  Print progress every 20 epochs
      if Epoch mod 20 = 0 then
         declare
            Current_Output : constant Float_Array := Get_Network_Output (Network);
            Current_Loss : constant Float := Calculate_Loss (Current_Output, Target_Data);
         begin
            Put_Line ("Epoch " & Epoch'Image &
                     ": Output = " & Current_Output (1)'Image &
                     ", Loss = " & Current_Loss'Image);
         end;
      end if;
   end loop;

   --  Show final results
   declare
      Final_Output : constant Float_Array := Get_Network_Output (Network);
      Final_Loss : constant Float := Calculate_Loss (Final_Output, Target_Data);
   begin
      Put_Line ("");
      Put_Line ("Final Results:");
      Put_Line ("Target: " & Target_Data (1)'Image);
      Put_Line ("Output: " & Final_Output (1)'Image);
      Put_Line ("Loss: " & Final_Loss'Image);
      Put_Line ("Error: " & Float'Image (abs (Final_Output (1) - Target_Data (1))));
   end;
end Example;