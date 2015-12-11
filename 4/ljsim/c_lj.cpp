#include <cmath>
#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

extern "C" {
  // a list of particle pairs within r_cut+skin
  vector<int> verlet_list;
  int N;
  double L;
  double rcut;
  double shift;

  void c_set_globals(double _L, int _N, double _rcut, double _shift) {
    N = _N;
    L = _L;
    rcut = _rcut;
    shift = _shift;
  }

  // compute the minimum image of particle i and j
  void minimum_image(double *x, int i, int j, double rij[3]) {
    rij[0] = x[j] - x[i];
    rij[1] = x[j+N] - x[i+N];
    rij[2] = x[j+2*N] - x[i+2*N];

    rij[0] -= rint(rij[0]/L)*L;
    rij[1] -= rint(rij[1]/L)*L;
    rij[2] -= rint(rij[2]/L)*L;
  }

  void compute_lj_force(double rij[3], double fij[3]) {
    double r2 = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2];
    if (r2 < rcut*rcut) {
      double fac = 4.0 * (12.0 * pow(r2, -6.5) - 6.0 * pow(r2, -3.5)) / sqrt(r2);
      fij[0] = fac*rij[0];
      fij[1] = fac*rij[1];
      fij[2] = fac*rij[2];
    } else {
      fij[0] = 0.0;
      fij[1] = 0.0;
      fij[2] = 0.0;
    }
  }
  
  double compute_lj_potential(double rij[3]) {
    double r2 = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2];
    if (r2 < rcut*rcut) {
      return 4.0 * (pow(r2, -6) - pow(r2, -3)) - shift;
    } else {
      return 0.0;
    }
  }

  void c_compute_forces(double *x, double *f) {
    double rij[3];
    double fij[3];

    // set all forces to zero
    for (int i = 0; i < 3*N; i++)
      f[i] = 0.0;

    // for (int i = 0; i < N; i++)
    //   for (int j = 0; j < N; j++) {

    // add up the forces
    vector<int>::iterator it = verlet_list.begin();
    vector<int>::iterator end = verlet_list.end();
    while (it != end) {
      int i = *it;
      ++it;
      int j = *it;
      ++it;
        if (i!=j){
      minimum_image(x, i, j, rij);
      compute_lj_force(rij, fij);

      f[i] -= fij[0];
      f[i+N] -= fij[1];
      f[i+2*N] -= fij[2];
      
      f[j] += fij[0];
      f[j+N] += fij[1];
      f[j+2*N] += fij[2];
    }
      }
  }

  double c_compute_energy(double *x, double *v, double* E_pot, double* E_kin) {
    double rij[3];
    *E_pot = 0.0;
    *E_kin = 0.0;

    // for (int i = 1; i < N; i++)
    //   for (int j = 0; j < i; j++) {

    // add up potential energy
    vector<int>::iterator it = verlet_list.begin();
    vector<int>::iterator end = verlet_list.end();
    while (it != end) {
      int i = *it;
      ++it;
      int j = *it;
      ++it;
      
      minimum_image(x, i, j, rij);
      *E_pot += compute_lj_potential(rij);
    }

    // add up kinetic energy
    for (int i = 0; i < N; i++) {
      *E_kin += 0.5*(v[i]*v[i] + v[i+N]*v[i+N] + v[i+2*N]*v[i+2*N]);
    }

    return *E_pot + *E_kin;
  }

  void c_compute_distances(double *x, double *rs) {
    double rij[3];
    
    int k = 0;
    for (int i = 1; i < N; i++)
      for (int j = 0; j < i; j++) {
        minimum_image(x, i, j, rij);
        rs[k] = sqrt(rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]);
        k++;
      }
  }

  // a list of the neighbor cells that need to be checked for
  // interaction partners
  const int neighbor_cell[39] = {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
    -1, 0, 1,
    0, -1, 1,
    0, 1, 1,
    1, 0, 1,
    -1, 1, 0,
    1, 1, 0,
    -1, 1, 1,
    1, -1, 1,
    1, 1, 1,
    -1, -1, 1,
  };


  // call this whenever a particle has moved further than skin
  void c_rebuild_neighbor_lists(double *x, const double vlsize) {
    // number of cells per side
    const int n = (int)floor(L/vlsize);
    const double rcut2 = vlsize*vlsize;

    // empty old verlet list
    verlet_list.resize(0);

    if (n > 3) {
      // inverse cell size
      const double li = n/L;
      // number of cells
      const int Nc = n*n*n;

      // create Nc empty cell lists
      vector < vector < int > > cell_list;
      cell_list.resize(Nc);

      // sort all particles into their cell list
      for (int i = 0; i < N; i++) {
        // x,y,z index of the cell
        int cx = int(floor(x[i] * li)) % n;
        if (cx < 0) cx += n;
        int cy = int(floor(x[i+N] * li)) % n;
        if (cy < 0) cy += n;
        int cz = int(floor(x[i+2*N] * li)) % n;
        if (cz < 0) cz += n;
        // linear index of the cell
        int cix = cx + cy*n + cz*n*n;
        // put the particle on the list
        cell_list[cix].push_back(i);
      }

      int sum = 0;
      for (int i = 0; i < Nc; i++) {
        sum += cell_list[i].size();
      }

      // now loop over neighboring cells to generate Verlet list
      for (int c0x = 0; c0x < n; c0x++) {
        for (int c0y = 0; c0y < n; c0y++) {
          for (int c0z = 0; c0z < n; c0z++) {
            // position of the origin box: c0x, c0y, c0z
            // index of the origin box
            int c0ix = c0x + c0y*n + c0z*n*n;

            // find the pairs within the cell itself
            vector<int>::iterator it0 = cell_list[c0ix].begin();
            vector<int>::iterator end0 = cell_list[c0ix].end();
            for (; it0 != end0; ++it0) {
              vector<int>::iterator it1 = it0;
              it1++;
              for (; it1 != end0; ++it1) {
                double rij[3];
                minimum_image(x, *it0, *it1, rij);
                if (rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2] < rcut2) {
                  verlet_list.push_back(*it0);
                  verlet_list.push_back(*it1);
                }
              }
            }

            // loop over the 13 neighboring cells 
            for (int i = 0; i < 39; i += 3) {
              // position of the neighbor box
              // heed pbc
              int c1x = (c0x+neighbor_cell[i]) % n;
              if (c1x < 0) c1x += n;
              int c1y = (c0y+neighbor_cell[i+1]) % n;
              if (c1y < 0) c1y += n;
              int c1z = (c0z+neighbor_cell[i+2]) % n;
              if (c1z < 0) c1z += n;
              // index of the neighbor box
              const int c1ix = c1x + c1y*n + c1z*n*n;
                  
              // loop over particle pairs in the cells
              vector<int>::iterator it0 = cell_list[c0ix].begin();
              vector<int>::iterator end0 = cell_list[c0ix].end();
              for (; it0 != end0; ++it0) {
                vector<int>::iterator it1 = cell_list[c1ix].begin();
                vector<int>::iterator end1 = cell_list[c1ix].end();
                for (; it1 != end1; ++it1) {
                  double rij[3];
                  minimum_image(x, *it0, *it1, rij);
                  if (rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2] < rcut2) {
                    verlet_list.push_back(*it0);
                    verlet_list.push_back(*it1);
                  }
                }
              }
            }
          }
        }
      }

    } else {
      // if n <= 3
      // build the verlet list without cell lists
      for (int i = 1; i < N; i++) {
        for (int j = 0; j < i; j++) {
          double rij[3];
          minimum_image(x, i, j, rij);
          if (rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2] < rcut2) {
            verlet_list.push_back(i);
            verlet_list.push_back(j);
          }
        }
      }
    }

    // checks
    // cout << "Number of pairs in VL: " << (verlet_list.size()/2) << endl;
  }
}
