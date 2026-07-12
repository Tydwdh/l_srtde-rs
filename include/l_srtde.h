#ifndef L_SRTDE_H
#define L_SRTDE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    LSRTDE_OK = 0,
    LSRTDE_NULL_CONFIG = 1,
    LSRTDE_NULL_BOUNDS = 2,
    LSRTDE_NULL_CALLBACK = 3,
    LSRTDE_NULL_OUTPUT = 4,
    LSRTDE_INVALID_DIMENSION = 5,
    LSRTDE_POPULATION_OVERFLOW = 6,
    LSRTDE_POPULATION_TOO_SMALL = 7,
    LSRTDE_INVALID_BOUNDS = 8,
    LSRTDE_CALLBACK_ERROR = 9,
    LSRTDE_NONFINITE_FITNESS = 10
};

/*
 * The input/output pointers are valid only during the callback. The callback
 * must write point_count finite values and return 0 on success.
 */
typedef int32_t (*lsrtde_evaluate_batch_fn)(
    const double *points,
    size_t point_count,
    size_t dim,
    double *fitness_out,
    void *user_data
);

typedef struct {
    size_t dim;
    const double *lower_bounds;
    const double *upper_bounds;
    size_t max_evaluations;
    size_t memory_size;
    size_t pop_size_multiplier;
    uint64_t seed;
    uint8_t use_seed;
} lsrtde_config;

/*
 * config->lower_bounds and config->upper_bounds each contain config->dim
 * elements. best_genome_out contains config->dim writable doubles and
 * best_fitness_out points to one writable double. All pointers must be valid
 * for the duration of this call.
 */
int32_t lsrtde_minimize(
    const lsrtde_config *config,
    lsrtde_evaluate_batch_fn evaluate_batch,
    void *user_data,
    double *best_genome_out,
    double *best_fitness_out
);

const char *lsrtde_error_message(int32_t code);

#ifdef __cplusplus
}
#endif

#endif
