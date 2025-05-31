import numpy as np
import pandas as pd
import os

class MotionConverter:
    def __init__(self, motion_file):
        current_path = os.path.dirname(os.path.abspath(__file__))
        self.motion_file = os.path.join(current_path, motion_file)
        self.load_motion()

    def load_motion(self):
        # check file extension
        if self.motion_file.endswith(".csv"):
            # load csv file
            self.motion_data: pd.DataFrame = pd.read_csv(self.motion_file)
        elif self.motion_file.endswith(".xlsx"):
            # load xlsx file
            self.motion_data: pd.DataFrame = pd.read_excel(self.motion_file)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

    def convert(self, output_file):
        if output_file.endswith(".npz"):
            self._convert_to_npz(output_file)
        elif output_file.endswith(".csv"):
            raise ValueError("Output file format .csv is not supported. Please use .npz.")

    def _convert_to_npz(self, output_file):
        # list all headers
        headers = self.motion_data.columns.tolist()

        # convert all columns to numpy arrays
        dof_positions = self.motion_data[headers[0:3]].to_numpy()
        # q_hip = q_hip - pi/2; q_knee = q_knee +pi/2
        dof_positions[:, 1] = dof_positions[:, 1] - np.pi / 2
        dof_positions[:, 2] = dof_positions[:, 2] + np.pi / 2

        dof_velocities = self.motion_data[headers[3:6]].to_numpy()
        dof_currents = self.motion_data[headers[6:]].to_numpy()
        data_dict = {
            "fps": np.array(50),
            "dof_names": np.array(["LFJ_scap", "LFJ_hip", "LFJ_knee"]),
            "dof_positions": dof_positions,
            "dof_velocities": dof_velocities,
            "dof_currents": dof_currents,
        }

        # save to npz file
        current_path = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(current_path, output_file)
        np.savez_compressed(output_file, **data_dict)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Csv motion file")
    parser.add_argument("--output_file", type=str, required=True, help="Output npz name")
    args, _ = parser.parse_known_args()

    motion_converter = MotionConverter(args.input_file)
    motion_converter.convert(args.output_file)