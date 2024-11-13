#pragma once

void collect_magnet(double x, double y, double z);

void compute_magnet_calibration(double **hard_iron, double **soft_iron);

void reset_magnet_calibration();

void magnet_align_with_gravity(double mag[3], double accel[3], double mag_result[3]);