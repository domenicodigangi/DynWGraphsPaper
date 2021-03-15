# SDERGM revision
## Practical To Do
- Check that results of estimates on eMid from pythorch make sense. 
    - they don't . 
    - check that at least the single snapshot estimates, do. They don't
    - Comparing the single snapshots with julia version to spot differences and fix them in Py
    - solve the SS estimate issue SOLVED
        - look at single snapshot estimates with new starting points and new value for zeros THE SS look good
         
- Look at filtered time varying parameters.
    - Some parameters explode to negative values. Their unconditional mean is very negative because 
        because the degrees are zero after a while (or since the beginning). I need to find a way to avoid this
            - first understand clearly why this does not happen in julia and check that we are actually rescaling. 
                In python was the unconditional mean of the zero degs par that was very negative. The issue was only 
                graphical. Capping the most negative W parameters did not change the loglikelihood.
            - Think of a way to limit negative values already in the SD filter that does not change the likelihood
            - pox 1 set a lower bound to the Uncond means (DONE WORKING)
            - pox 2 adjust the parameters of those with zero degrees at each iteration
- Compute hessian of the log likelihood.
    - check why the gradient is so large and fix also the hessian
    - re running the simulations to understand why the gradient is not small at estimates. Likely a simple bug
        - there is no guarantee that the gradient should be zero...
        - is there a parametrization (link function or static par transf) that makes the optimization convex?? 
        - To be launched TEST ON CONSISTENCY OF THE JOINT ESTIMATE FOR MULTI PARAMETERS AND SEQUENCEM OF TWO STEPS
        - WRITE SD DGP FOR BINARY CASE AND LAUNCH TEST ON ESTIMATES WITH DIFFEERENT METHODS. Running
        - Add test using adam with adaptive learning rate
-Now that the hessian is available!!! compute a first test for the confidence intervals

- Compute confidence intervals.
- Discuss results.
