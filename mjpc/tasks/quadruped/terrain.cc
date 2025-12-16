#include <algorithm>
#include <cmath>
#include <limits>

#include "terrain.h"


namespace mjpc {
  
Terrain::Terrain(const mjModel* model) { Initialize(model); }
  
void Terrain::Initialize(const mjModel* model) {
    
    geom_id   = mj_name2id(model, mjOBJ_GEOM, "terrain");
    hfield_id = mj_name2id(model, mjOBJ_HFIELD, "hf133");

    if (geom_id < 0 || hfield_id < 0) {
        mju_error("ERROR: Terrain::Initialize - geom_id = %d, hfield_id = %d\n", geom_id, hfield_id);
    }

    nrow = model->hfield_nrow[hfield_id];
    ncol = model->hfield_ncol[hfield_id];
    adr  = model->hfield_adr[hfield_id];

    const mjtNum* size = model->hfield_size + 4 * hfield_id; // (x, y, z_top, _)  (nhfield x 4)

    sx = size[0];
    sy = size[1];
    sz = size[2];

    if(nrow <= 1 || ncol <= 1) {
        mju_error("ERROR: Terrain::Initialize - nrow = %d, ncol = %d\n", nrow, ncol);
    }

    dx = (2.0 * sx) / (ncol - 1);  
    dy = (2.0 * sy) / (nrow - 1);

    H = model->hfield_data + adr;
}
    
void Terrain::GetVertexNormal(int r, int c, float* n) const {
    
    auto h = [&](int r, int c) -> double {
        r = mjMAX(0, mjMIN(r, nrow - 1));
        c = mjMAX(0, mjMIN(c, ncol - 1));
        return static_cast<double>(H[r * ncol + c]) * sz;
    };

    double hx;
    if (c == 0) {
        hx = (h(r, 1) - h(r, 0)) / dx;
    } else if (c == ncol - 1) {
        hx = (h(r, ncol - 1) - h(r, ncol - 2)) / dx;
    } else {
        hx = (h(r, c + 1) - h(r, c - 1)) / (2 * dx);
    }

    double hy;
    if (r == 0) {
        hy = (h(1, c) - h(0, c)) / dy;
    } else if (r == nrow - 1) {
        hy = (h(nrow - 1, c) - h(nrow - 2, c)) / dy;
    } else {
        hy = (h(r + 1, c) - h(r - 1, c)) / (2 * dy);
    }

    double inv = 1.0 / mju_sqrt(hx * hx + hy * hy + 1.0 + 1e-30);
    n[0] = static_cast<float>(-hx * inv);
    n[1] = static_cast<float>(-hy * inv);
    n[2] = static_cast<float>(inv);
}
  
/* Bilinear height query: x and y in [-sx, sx] and [-sy, sy]. */
void Terrain::GetHeightFromLocal(double x, double y, double& z) const {

    BilinearParams p = LocalToGrid(x, y);

    double h00 = H[p.y0 * ncol + p.x0] * sz;
    double h10 = H[p.y0 * ncol + p.x1] * sz;
    double h01 = H[p.y1 * ncol + p.x0] * sz;
    double h11 = H[p.y1 * ncol + p.x1] * sz;

    double h0 = (1.0 - p.tx) * h00 + p.tx * h10;
    double h1 = (1.0 - p.tx) * h01 + p.tx * h11;

    z = (1.0 - p.ty) * h0 + p.ty * h1;
}
  
/* Bilinear normal query: x and y in [-sx, sx] and [-sy, sy]. */
void Terrain::GetNormalFromLocal(double x, double y, double n[3]) const {

    BilinearParams p = LocalToGrid(x, y);

    float n00[3], n10[3], n01[3], n11[3];
    GetVertexNormal(p.y0, p.x0, n00);
    GetVertexNormal(p.y0, p.x1, n10);
    GetVertexNormal(p.y1, p.x0, n01);
    GetVertexNormal(p.y1, p.x1, n11);

    double nx0 = (1.0 - p.tx) * n00[0] + p.tx * n10[0];
    double ny0 = (1.0 - p.tx) * n00[1] + p.tx * n10[1];
    double nz0 = (1.0 - p.tx) * n00[2] + p.tx * n10[2];

    double nx1 = (1.0 - p.tx) * n01[0] + p.tx * n11[0];
    double ny1 = (1.0 - p.tx) * n01[1] + p.tx * n11[1];
    double nz1 = (1.0 - p.tx) * n01[2] + p.tx * n11[2];

    double nx = (1.0 - p.ty) * nx0 + p.ty * nx1;
    double ny = (1.0 - p.ty) * ny0 + p.ty * ny1;
    double nz = (1.0 - p.ty) * nz0 + p.ty * nz1;

    // normalize to guard against interpolation drift
    double inv = 1.0 / mju_sqrt(nx * nx + ny * ny + nz * nz + 1e-30);
    n[0] = nx * inv;
    n[1] = ny * inv;
    n[2] = nz * inv;
}
  
void Terrain::GetHeightFromWorld(const mjData* data, double x, double y, double& z_world) const {

    const double* R;
    const double* t;
    double p_local[3];
    WorldToLocal(data, x, y, R, t, p_local);

    double z_local;
    GetHeightFromLocal(p_local[0], p_local[1], z_local);

    double s_local[3] = {p_local[0], p_local[1], z_local};
    double s_world[3];
    mju_mulMatVec(s_world, R, s_local, 3, 3);
    z_world = s_world[2] + t[2];
}
  
void Terrain::GetNormalFromWorld(const mjData* data, double x, double y, double n[3]) const {
    
    const double* R;
    const double* t;
    double p_local[3];
    WorldToLocal(data, x, y, R, t, p_local);
    
    double n_local[3];
    GetNormalFromLocal(p_local[0], p_local[1], n_local);
    
    mju_mulMatVec(n, R, n_local, 3, 3);
    double inv = 1.0 / mju_sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2] + 1e-30);
    n[0] *= inv;
    n[1] *= inv;
    n[2] *= inv;
  }
  
void Terrain::GetPatchFeatures(const mjData* data, double x, double y,
                               PatchFeatures& features,
                               double patch_radius) const {
    const double* R;
    const double* t;
    double p_local[3];
    WorldToLocal(data, x, y, R, t, p_local);

    // map to nearest grid index using local coords
    double u = (p_local[0] + sx) / dx;
    double v = (p_local[1] + sy) / dy;
    int c = static_cast<int>(mju_floor(u));
    int r = static_cast<int>(mju_floor(v));

    double sum[3] = {0.0, 0.0, 0.0};
    double sum_outer[9];
    mju_zero(sum_outer, 9);
    double zmin = std::numeric_limits<double>::infinity();
    double zmax = -std::numeric_limits<double>::infinity();
    int samples = 0;

    int kr = std::max(1, static_cast<int>(std::ceil(patch_radius / dy)));
    int kc = std::max(1, static_cast<int>(std::ceil(patch_radius / dx)));

    // single-pass accumulation over 3x3 neighborhood
    for (int dr = -kr; dr <= kr; ++dr) {
        for (int dc = -kc; dc <= kc; ++dc) {
            int rr = mjMAX(0, mjMIN(r + dr, nrow - 1));
            int cc = mjMAX(0, mjMIN(c + dc, ncol - 1));

            double px = -sx + cc * dx;
            double py = -sy + rr * dy;
            double pz = static_cast<double>(H[rr * ncol + cc]) * sz;

            double p[3] = {px, py, pz};
            mju_add3(sum, sum, p);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    sum_outer[i + 3 * j] += p[i] * p[j];
                }
            }

            zmin = std::min(zmin, pz);
            zmax = std::max(zmax, pz);
            ++samples;
        }
    }

    double inv_n = samples > 0 ? 1.0 / static_cast<double>(samples) : 0.0;
    double centroid[3];
    mju_scl3(centroid, sum, inv_n);

    double cov[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cov[i + 3 * j] = (sum_outer[i + 3 * j] * inv_n) - centroid[i] * centroid[j];
        }
    }

    double eigval[3];
    double eigvec[9];
    double quat[4];
    mju_eig3(eigval, eigvec, quat, cov);

    // convert centroid to world frame
    double centroid_world[3];
    mju_mulMatVec(centroid_world, R, centroid, 3, 3);
    mju_addTo3(centroid_world, t);
    mju_copy3(features.centroid, centroid_world);

    features.roughness = mju_sqrt(mjMAX(eigval[2], 0.0));

    // normal to world frame
    double n_local[3] = {
        eigvec[0 + 3 * 2],
        eigvec[1 + 3 * 2],
        eigvec[2 + 3 * 2]
    };
    if (n_local[2] < 0.0) {
        mju_scl3(n_local, n_local, -1.0);
    }
    mju_mulMatVec(features.normal, R, n_local, 3, 3);
    double inv = 1.0 / mju_sqrt(features.normal[0] * features.normal[0] +
                                features.normal[1] * features.normal[1] +
                                features.normal[2] * features.normal[2] + 1e-30);
    features.normal[0] *= inv;
    features.normal[1] *= inv;
    features.normal[2] *= inv;

    features.step_height = zmax - zmin;
    features.max_height = zmax;
}

bool Terrain::IsSafe(const mjData* data, double x, double y) const {
    PatchFeatures features{};
    GetPatchFeatures(data, x, y, features);
    return (features.roughness <= kMaxRoughness) &&
           (features.normal[2] >= kMinNormalZ) &&
           (features.step_height <= kMaxStepHeight);
  }
  
  
/* ------------ Auxiliary Functions ------------ */
void Terrain::WorldToLocal(const mjData* data, double x, double y,
                          const double*& R, const double*& t,
                          double p_local[3]) const {
    R = data->geom_xmat + 9 * geom_id;
    t = data->geom_xpos + 3 * geom_id;
    double p_world[3] = {x - t[0], y - t[1], 0.0};
    mju_mulMatTVec(p_local, R, p_world, 3, 3);
}

Terrain::BilinearParams Terrain::LocalToGrid(double x, double y) const {
    double u = (x + sx) / dx;
    double v = (y + sy) / dy;
    
    int x0 = (int)mju_floor(u);
    int y0 = (int)mju_floor(v);
    
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    double tx = u - x0;
    double ty = v - y0;
    
    x0 = mjMAX(0, mjMIN(x0, ncol - 1));
    x1 = mjMAX(0, mjMIN(x1, ncol - 1));
    y0 = mjMAX(0, mjMIN(y0, nrow - 1));
    y1 = mjMAX(0, mjMIN(y1, nrow - 1));
    
    return {x0, x1, y0, y1, tx, ty};
  }
  
}
