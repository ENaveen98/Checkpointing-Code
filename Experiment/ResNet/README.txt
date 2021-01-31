1. Input the desired Neural Network in Step1_Memory_Profiler.py script in same format as the example provided and run it from the terminal.
2. This would generate a memory_profiler.log file which would contain information about the memory consumption by each layer of the Neural Network.
3. Run the cells inside Step2_Optimal_Checkpointing_Configuration.ipynb jupyter notebook to parse the log file and obtain the optimal checkpointing configuration.
4. The final output in the notebook would be the allocation of layers into different segments after checkpointing which would approximately consume the least amount of memory while training.
5. If you need the allocation of layers into different segments for an alternate checkpointing configuration provide the indices of checkpoint layers in a list. See example provided in the last cell.
6. Implement this allocation in Step3_Checkpointing_Neural_Network.py script using the sample format provided and run it from the terminal to train the Network with reduced memory consumption.

NOTE: For practical use cases consider using a lighter version of the pytorch framework or employ other similar steps as the current framework utilizes a fairly large amount of memory just to load the package.
NOTE: The required python packages are listed in requirements.txt file and can be installed by using the following command in Windows (or a similar one for other Operating Systems): pip install -r requirements.txt
