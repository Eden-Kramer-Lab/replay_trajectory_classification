from dataclasses import dataclass


@dataclass
class UniformInitialConditions:
    environment_names: tuple

    def make_initial_conditions(self, environments):
        n_total_place_bins = 0
        initial_conditions = []

        for environment_name in self.environment_names:
            is_track_interior = (environments[environment_name]
                                 .is_track_interior_
                                 .ravel(order='F'))
            n_place_bins = is_track_interior.sum()
            n_total_place_bins += n_place_bins
            initial_conditions.append(is_track_interior)

        return [ic / n_total_place_bins for ic in initial_conditions]
