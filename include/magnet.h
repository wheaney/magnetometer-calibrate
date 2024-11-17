#pragma once


#include <stdint.h>

void collect_magnet(double x, double y, double z, uint32_t timestamp_ms);

void compute_magnet_calibration(double (*hard_iron)[3], double (*soft_iron)[9]);

void reset_magnet_calibration();

void magnet_align_with_gravity(double mag[3], double accel[3], double mag_result[3]);