Algorithm 1: Proposed PSO-MEM Algorithm
Input:
The DNA sequence dataset.
Output:
Classify 4mC and non-4mC
  1. Begin
  2. Remove the redundant sequences.
  3. Extract the semantic features from the gene sequences.
  4. Fill the missing value with the mean of the column and remove the duplicate rows.
  5. Transform features or variables to a specific range, typically between 0 and 1.
  6. If total number of  total number of   then
  7.          Randomly select the equal number of   and  
  8.  Else 
  9.                  [Where   =1,2,3,…,8]
 10.  End If
 11.  Split the dataset into training   and testing   set. Where  > . The ratio between 
        and   of each data set is the same as in the entire dataset .
 12. Select the best subset of features for Random Forest (RF), Support Vector Machine (SVM), and Gradient 
       Boosting (GB), and K-Nearest Neighbor classifier using Particle Swarm Optimization.
 13. Tuning the base classifiers with numerous hyper-parameters.
 14. Apply training set to fit RF, SVM, GB with optimum features.
 15. Integrate the base classifiers.
 16. Predict new data.
 17. For  =2 to n do
 18. 	If TN and TP  for RF, SVM, KNN then 
 19.                	  =    + Number of TP, TN in layer 
 20.                	 = Number of FP, FN in layer           [Where   = layer number]
 21.                	 Repeat Step 13 to 16.
 22.	Else 
 23.                       Apply KNN to find the optimal number of nearest neighborhood data points.
 24.		Predict new data.
		if TN and TP  
 25.			Repeat Step 19 to 20 and 23 to 24.	
 26.           	Else
 27.                		Calculate total accuracy using the following formula:
 28.                        	    		     [Where   = 1, 2, 3,……., n]
 29.        End If
 30. End For
 31. Stop 

![Proposed Model(PSO-MEM)](https://github.com/rajib1346/PSO-MEM/assets/26224493/969a2aff-f7cd-43f8-9fa3-9ecb21f02d30)
