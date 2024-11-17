#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>

/**
 * Used MagCal C# implementation as reference: https://github.com/hightower70/MagCal/blob/d0abf0fa6d08cb0cc8bd4dd458003289ebcecef9/Program.cs
 * Generated almost entirely by o1-preview.
 */

// Global variables
static double S_accum[10][10] = {0};  // Accumulated S matrix
static int sample_count = 0;

// Function to accumulate magnetometer samples
void collect_magnet(double x, double y, double z, uint32_t timestamp_ms) {
#ifndef NDEBUG
    static int last_sample_ms = 0;

    if (last_sample_ms != 0) {
        int delta_ms = timestamp_ms - last_sample_ms;
        // extract for use with a program like Magneto, for example: 
        //      grep "Mag sample:" log_file.txt | awk -F'\t' '{print $2 "\t" $3 "\t" $4}' > mag.txt
        printf("Mag sample:\t%f\t%f\t%f\t%d\n", x, y, z, delta_ms);
    }
    last_sample_ms = timestamp_ms;
#endif

    double D[10];
    D[0] = x * x;
    D[1] = y * y;
    D[2] = z * z;
    D[3] = 2.0 * y * z;
    D[4] = 2.0 * x * z;
    D[5] = 2.0 * x * y;
    D[6] = 2.0 * x;
    D[7] = 2.0 * y;
    D[8] = 2.0 * z;
    D[9] = 1.0;

    // Update S matrix
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j <= i; j++) {  // Since S is symmetric
            S_accum[i][j] += D[i] * D[j];
            if (i != j) {
                S_accum[j][i] = S_accum[i][j];
            }
        }
    }
    sample_count++;
}

// Function to compute the pseudoinverse of a matrix using SVD
static void compute_pseudoinverse(gsl_matrix *A, gsl_matrix *A_pinv) {
    int m = A->size1;
    int n = A->size2;
    int i, j;

    gsl_matrix *U = gsl_matrix_alloc(m, n);
    gsl_matrix *V = gsl_matrix_alloc(n, n);
    gsl_vector *S = gsl_vector_alloc(n);
    gsl_vector *work = gsl_vector_alloc(n);

    gsl_matrix_memcpy(U, A);
    gsl_linalg_SV_decomp(U, V, S, work);

    // Compute Sinv (reciprocal of singular values)
    gsl_matrix *Sinv = gsl_matrix_alloc(n, n);
    gsl_matrix_set_zero(Sinv);
    for (i = 0; i < n; i++) {
        double s = gsl_vector_get(S, i);
        if (s > 1e-8)
            gsl_matrix_set(Sinv, i, i, 1.0 / s);
    }

    // Compute pseudoinverse: A_pinv = V * Sinv * U^T
    gsl_matrix *temp = gsl_matrix_alloc(n, m);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Sinv, U, 0.0, temp);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, V, temp, 0.0, A_pinv);

    // Free allocated memory
    gsl_matrix_free(U);
    gsl_matrix_free(V);
    gsl_vector_free(S);
    gsl_vector_free(work);
    gsl_matrix_free(Sinv);
    gsl_matrix_free(temp);
}

// Function to invert a matrix
static void invert_matrix(gsl_matrix *A, gsl_matrix *A_inv) {
    int n = A->size1;
    int s;

    gsl_permutation *p = gsl_permutation_alloc(n);
    gsl_matrix *A_copy = gsl_matrix_alloc(n, n);
    gsl_matrix_memcpy(A_copy, A);

    gsl_linalg_LU_decomp(A_copy, p, &s);
    gsl_linalg_LU_invert(A_copy, p, A_inv);

    gsl_permutation_free(p);
    gsl_matrix_free(A_copy);
}

// Function to compute eigenvalues and eigenvectors of a real nonsymmetric matrix
static void compute_eigen(gsl_matrix *A, gsl_vector_complex *eval, gsl_matrix_complex *evec) {
    int n = A->size1;

    gsl_eigen_nonsymmv_workspace *w = gsl_eigen_nonsymmv_alloc(n);
    gsl_matrix *A_copy = gsl_matrix_alloc(n, n);
    gsl_matrix_memcpy(A_copy, A);

    gsl_eigen_nonsymmv(A_copy, eval, evec, w);

    gsl_eigen_nonsymmv_free(w);
    gsl_matrix_free(A_copy);
}

// Q Li Ellipsoid Fitting algorithm
// Used FindFit4 MATLAB function as reference: https://www.mathworks.com/matlabcentral/fileexchange/23377-ellipsoid-fitting
static void FindFit4(gsl_matrix *S, gsl_vector *v) {
    int i;

    // Create submatrices of S
    gsl_matrix_const_view S11_view = gsl_matrix_const_submatrix(S, 0, 0, 6, 6);
    gsl_matrix_const_view S12_view = gsl_matrix_const_submatrix(S, 0, 6, 6, 4);
    gsl_matrix_const_view S12t_view = gsl_matrix_const_submatrix(S, 6, 0, 4, 6);
    gsl_matrix_const_view S22_view = gsl_matrix_const_submatrix(S, 6, 6, 4, 4);

    gsl_matrix *S11 = gsl_matrix_alloc(6, 6);
    gsl_matrix *S12 = gsl_matrix_alloc(6, 4);
    gsl_matrix *S12t = gsl_matrix_alloc(4, 6);
    gsl_matrix *S22 = gsl_matrix_alloc(4, 4);

    gsl_matrix_memcpy(S11, &S11_view.matrix);
    gsl_matrix_memcpy(S12, &S12_view.matrix);
    gsl_matrix_memcpy(S12t, &S12t_view.matrix);
    gsl_matrix_memcpy(S22, &S22_view.matrix);

    // Compute pseudoinverse of S22
    gsl_matrix *S22_pinv = gsl_matrix_alloc(4, 4);
    compute_pseudoinverse(S22, S22_pinv);

    // Compute SS = S11 - S12 * S22_pinv * S12t
    gsl_matrix *temp1 = gsl_matrix_alloc(6, 4);
    gsl_matrix *temp2 = gsl_matrix_alloc(6, 6);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, S12, S22_pinv, 0.0, temp1);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, temp1, S12t, 0.0, temp2);

    gsl_matrix *SS = gsl_matrix_alloc(6, 6);
    gsl_matrix_memcpy(SS, S11);
    gsl_matrix_sub(SS, temp2);

    // Create constraint matrix Co
    double Co_data[6][6] = {
        { -1.0,  1.0,  1.0,  0.0,  0.0,  0.0 },
        {  1.0, -1.0,  1.0,  0.0,  0.0,  0.0 },
        {  1.0,  1.0, -1.0,  0.0,  0.0,  0.0 },
        {  0.0,  0.0,  0.0, -4.0,  0.0,  0.0 },
        {  0.0,  0.0,  0.0,  0.0, -4.0,  0.0 },
        {  0.0,  0.0,  0.0,  0.0,  0.0, -4.0 }
    };

    gsl_matrix *Co = gsl_matrix_alloc(6, 6);
    for (i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            gsl_matrix_set(Co, i, j, Co_data[i][j]);

    // Compute inverse of Co
    gsl_matrix *C = gsl_matrix_alloc(6, 6);
    invert_matrix(Co, C);

    // Calculate E = C * SS
    gsl_matrix *E = gsl_matrix_alloc(6, 6);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, C, SS, 0.0, E);

    // Compute eigenvalues and eigenvectors of E
    gsl_vector_complex *eval = gsl_vector_complex_alloc(6);
    gsl_matrix_complex *evec = gsl_matrix_complex_alloc(6, 6);
    compute_eigen(E, eval, evec);

    // Find the index of the largest positive real eigenvalue
    int index = -1;
    double maxval = -GSL_POSINF;
    for (i = 0; i < 6; i++) {
        gsl_complex eigenvalue = gsl_vector_complex_get(eval, i);
        double real_part = GSL_REAL(eigenvalue);
        if (real_part > maxval) {
            maxval = real_part;
            index = i;
        }
    }

    // Extract the associated eigenvector v1
    gsl_vector_complex_view evec_view = gsl_matrix_complex_column(evec, index);
    gsl_vector *v1 = gsl_vector_alloc(6);
    for (i = 0; i < 6; i++) {
        gsl_complex z = gsl_vector_complex_get(&evec_view.vector, i);
        gsl_vector_set(v1, i, GSL_REAL(z)); // Use real part
    }

    // Check sign of eigenvector v1
    if (gsl_vector_get(v1, 0) < 0.0) {
        gsl_vector_scale(v1, -1.0);
    }

    // Calculate v2 = (S22_pinv * S12t) * v1
    gsl_matrix *temp3 = gsl_matrix_alloc(4, 6);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, S22_pinv, S12t, 0.0, temp3);

    gsl_vector *v2 = gsl_vector_alloc(4);
    gsl_blas_dgemv(CblasNoTrans, 1.0, temp3, v1, 0.0, v2);

    // Construct vector v
    for (i = 0; i < 6; i++)
        gsl_vector_set(v, i, gsl_vector_get(v1, i));
    for (i = 0; i < 4; i++)
        gsl_vector_set(v, i + 6, -gsl_vector_get(v2, i));

    // Free allocated memory
    gsl_matrix_free(S11);
    gsl_matrix_free(S12);
    gsl_matrix_free(S12t);
    gsl_matrix_free(S22);
    gsl_matrix_free(S22_pinv);
    gsl_matrix_free(temp1);
    gsl_matrix_free(temp2);
    gsl_matrix_free(Co);
    gsl_matrix_free(C);
    gsl_matrix_free(E);
    gsl_vector_complex_free(eval);
    gsl_matrix_complex_free(evec);
    gsl_vector_free(v1);
    gsl_matrix_free(temp3);
    gsl_vector_free(v2);
}

// Function to generate calibration parameters based on current snapshot of magnetometer samples
void compute_magnet_calibration(double (*hard_iron)[3], double (*soft_iron)[9]) {
    gsl_matrix *S = gsl_matrix_calloc(10, 10);
    gsl_vector *v = gsl_vector_calloc(10);
    
    // Populate S matrix
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            gsl_matrix_set(S, i, j, S_accum[i][j]);
        }
    }

    // Find fitting
    FindFit4(S, v);

#ifndef NDEBUG
    printf("FindFit4 result:\n");
    for (int i = 0; i < 10; i++) {
        printf("\tv[%d] = %f\n", i, gsl_vector_get(v, i));
    }
#endif

    // Create matrix Q (3x3)
    gsl_matrix *Q = gsl_matrix_alloc(3, 3);

    gsl_matrix_set(Q, 0, 0, gsl_vector_get(v, 0)); // v[0]; // A
    gsl_matrix_set(Q, 0, 1, gsl_vector_get(v, 5)); // v[5]; // D
    gsl_matrix_set(Q, 0, 2, gsl_vector_get(v, 4)); // v[4]; // E
    gsl_matrix_set(Q, 1, 0, gsl_vector_get(v, 5)); // v[5]; // D
    gsl_matrix_set(Q, 1, 1, gsl_vector_get(v, 1)); // v[1]; // B
    gsl_matrix_set(Q, 1, 2, gsl_vector_get(v, 3)); // v[3]; // F
    gsl_matrix_set(Q, 2, 0, gsl_vector_get(v, 4)); // v[4]; // E
    gsl_matrix_set(Q, 2, 1, gsl_vector_get(v, 3)); // v[3]; // F
    gsl_matrix_set(Q, 2, 2, gsl_vector_get(v, 2)); // v[2]; // C

    // Create vector U (3x1)
    gsl_vector *U = gsl_vector_alloc(3);

    gsl_vector_set(U, 0, gsl_vector_get(v, 6)); // v[6]; // G
    gsl_vector_set(U, 1, gsl_vector_get(v, 7)); // v[7]; // H
    gsl_vector_set(U, 2, gsl_vector_get(v, 8)); // v[8]; // I

    // Invert Q to get Q_inv
    gsl_matrix *Q_inv = gsl_matrix_alloc(3, 3);
    invert_matrix(Q, Q_inv);

    // Calculate B = Q_inv * U
    gsl_vector *B = gsl_vector_alloc(3);
    gsl_blas_dgemv(CblasNoTrans, 1.0, Q_inv, U, 0.0, B);

    // Calculate combined bias: B = -B
    gsl_vector_scale(B, -1.0);

    // Calculate btqb = B^T * Q * B
    gsl_vector *temp_vec = gsl_vector_alloc(3);
    gsl_blas_dgemv(CblasNoTrans, 1.0, Q, B, 0.0, temp_vec);

    double btqb;
    gsl_blas_ddot(B, temp_vec, &btqb);

    gsl_vector_free(temp_vec);

    // Calculate hmb = sqrt(btqb - J)
    double J = gsl_vector_get(v, 9);
    double hmb = sqrt(btqb - J);

    // Calculate the square root of matrix Q using eigenvalue decomposition
    gsl_vector *eval = gsl_vector_alloc(3);
    gsl_matrix *evec = gsl_matrix_alloc(3, 3);

    gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(3);
    gsl_eigen_symmv(Q, eval, evec, w);
    gsl_eigen_symmv_free(w);

    // Normalize eigenvectors
    int i;
    for (i = 0; i < 3; i++) {
        gsl_vector_view vec_i = gsl_matrix_column(evec, i);
        double norm = gsl_blas_dnrm2(&vec_i.vector);
        gsl_vector_scale(&vec_i.vector, 1.0 / norm);
    }

    // Create Dz as a diagonal matrix with square roots of eigenvalues
    gsl_matrix *Dz = gsl_matrix_calloc(3, 3); // Initializes to zero
    for (i = 0; i < 3; i++) {
        double sqrt_eval = sqrt(gsl_vector_get(eval, i));
        gsl_matrix_set(Dz, i, i, sqrt_eval);
    }

    // Calculate SQ = evec * Dz * evec^T
    gsl_matrix *temp_matrix = gsl_matrix_alloc(3, 3);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, evec, Dz, 0.0, temp_matrix);

    gsl_matrix *evec_T = gsl_matrix_alloc(3, 3);
    gsl_matrix_transpose_memcpy(evec_T, evec);

    gsl_matrix *SQ = gsl_matrix_alloc(3, 3);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, temp_matrix, evec_T, 0.0, SQ);

    // Calculate A_inv = SQ * (hm / hmb)
    double hm = 0.569;
    double scale = hm / hmb;
    gsl_matrix_scale(SQ, scale);

    gsl_matrix *A = SQ;

    gsl_vector_view hard_iron_view = gsl_vector_view_array(*hard_iron, 3);
    gsl_vector_memcpy(&hard_iron_view.vector, B);
    
    gsl_matrix_view soft_iron_view = gsl_matrix_view_array(*soft_iron, 3, 3);
    gsl_matrix_memcpy(&soft_iron_view.matrix, A);

    gsl_vector_free(B);
    gsl_vector_free(U);
    gsl_vector_free(eval);
    gsl_matrix_free(evec);
    gsl_matrix_free(Q);
    gsl_matrix_free(Q_inv);
    gsl_matrix_free(Dz);
    gsl_matrix_free(SQ);
    gsl_vector_free(v);
    gsl_matrix_free(S);

#ifndef NDEBUG
    printf("\nHard Iron Offsets (Biases):\n");
    printf("\thx = %g\n", (*hard_iron)[0]);
    printf("\thy = %g\n", (*hard_iron)[1]);
    printf("\thz = %g\n", (*hard_iron)[2]);

    printf("\nSoft Iron Calibration Matrix (A-1):\n");
    printf("\t[%g, %g, %g]\n", (*soft_iron)[0], (*soft_iron)[1], (*soft_iron)[2]);
    printf("\t[%g, %g, %g]\n", (*soft_iron)[3], (*soft_iron)[4], (*soft_iron)[5]);
    printf("\t[%g, %g, %g]\n", (*soft_iron)[6], (*soft_iron)[7], (*soft_iron)[8]);
#endif
}

// Clear the accumulated magnetometer samples
void reset_magnet_calibration() {
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            S_accum[i][j] = 0;
        }
    }

    sample_count = 0;
}

// Projects the calibrated magnet reading onto the horizontal plane, using gravity as a reference.
// Useful for getting an accurate heading.
void magnet_align_with_gravity(double magnet[3], double accel[3], double result[3]) {
    // Normalize accel
    double mag_accel = sqrt(accel[0]*accel[0] + accel[1]*accel[1] + accel[2]*accel[2]);
    double u_accel[3] = { accel[0]/mag_accel, accel[1]/mag_accel, accel[2]/mag_accel };

    // Reference vector [0, 0, 1]
    double ref[3] = { 0.0, 0.0, 1.0 };

    // Compute dot product
    double dot = u_accel[0]*ref[0] + u_accel[1]*ref[1] + u_accel[2]*ref[2];
    if (dot > 1.0) dot = 1.0;
    if (dot < -1.0) dot = -1.0;

    // Compute angle
    double angle = acos(dot);

    // Compute rotation axis (cross product)
    double axis[3] = {
        u_accel[1]*ref[2] - u_accel[2]*ref[1],
        u_accel[2]*ref[0] - u_accel[0]*ref[2],
        u_accel[0]*ref[1] - u_accel[1]*ref[0]
    };

    // Compute magnitude of axis
    double mag_axis = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);

    if (mag_axis < 1e-8) {
        // Axis is zero vector (vectors are parallel or anti-parallel)
        if (dot > 0.9999) {
            // Vectors are aligned, no rotation needed
            result[0] = magnet[0];
            result[1] = magnet[1];
            result[2] = magnet[2];
            return;
        } else {
            // Vectors are opposite, rotate 180 degrees around arbitrary axis
            axis[0] = 1.0;
            axis[1] = 0.0;
            axis[2] = 0.0;
            mag_axis = 1.0;
            angle = M_PI;
        }
    }

    // Normalize rotation axis
    double u_axis[3] = { axis[0]/mag_axis, axis[1]/mag_axis, axis[2]/mag_axis };

    // Compute Rodrigues' rotation formula components
    double cos_angle = cos(angle);
    double sin_angle = sin(angle);
    double one_minus_cos = 1.0 - cos_angle;

    // Compute k x magnet (cross product)
    double k_cross_magnet[3] = {
        u_axis[1]*magnet[2] - u_axis[2]*magnet[1],
        u_axis[2]*magnet[0] - u_axis[0]*magnet[2],
        u_axis[0]*magnet[1] - u_axis[1]*magnet[0]
    };

    // Compute k ⋅ magnet (dot product)
    double k_dot_magnet = u_axis[0]*magnet[0] + u_axis[1]*magnet[1] + u_axis[2]*magnet[2];

    // Compute rotated vector
    result[0] = magnet[0]*cos_angle + k_cross_magnet[0]*sin_angle + u_axis[0]*k_dot_magnet*one_minus_cos;
    result[1] = magnet[1]*cos_angle + k_cross_magnet[1]*sin_angle + u_axis[1]*k_dot_magnet*one_minus_cos;
    result[2] = magnet[2]*cos_angle + k_cross_magnet[2]*sin_angle + u_axis[2]*k_dot_magnet*one_minus_cos;
}