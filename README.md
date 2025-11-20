The simulation data directory has the config needed for each simulation ran (a total of 4 simulations).
The simulations require:
  1. Have Smilei downloaded and available to run - https://github.com/SmileiPIC/Smilei
  2. Clone ALFP - https://github.com/kcharbo3/Arbitrary-Laser-Fields-for-PIC
  3. Swap out the `sim_config.py` files for the simulation you want to run. Use the `smilei.py` files to write the namelist (the namelist is in this file).
  4. Run the simulation.

Once the simulations have been run, you can use the `figure_notebook.ipynb` to generate the figures. Just replace the constants in the first cell that refer to directory paths with the correct paths.
