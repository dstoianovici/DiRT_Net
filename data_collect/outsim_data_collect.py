#!/usr/bin/env python3

import os, sys
import argparse
import socket
import struct

class OutSim_Data_Collect():
    def __init__(self, port):
        self._port = port
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.data_socket.bind(('127.0.0.1',self._port))
        self.read_data()
    
    def read_data(self):
        while True:
            data = self.data_socket.recv(512) # 64 Float64 values in extradata = 3
            if not data:
                break #Lost Comms

            # print(len(data))

            outsim_pack = struct.unpack('64f', data[0:256])

            # Curtesy of fab1701 https://steamcommunity.com/app/310560/discussions/0/481115363869500839/?l=norwegian
            _time = outsim_pack[0] # Time
            _lap_time = outsim_pack[1] # Time of Current Lap
            _lap_dist = outsim_pack[2] # Distance Driven on Current Lap
            _overall_dist = outsim_pack[3] # Driven Overall
            _pos_x = outsim_pack[4] # Position x
            _pos_y = outsim_pack[5] # Position y
            _pos_z = outsim_pack[6] # Position z
            _vel = outsim_pack[7] # Vehicle Velocity [m/s]
            _vel_x = outsim_pack[8] # Velocity x
            _vel_y = outsim_pack[9] # Velocity y
            _vel_z = outsim_pack[10] # Velocity z
            _roll_vec_x = outsim_pack[11] # Roll Vector x
            _roll_vec_y = outsim_pack[12] # Roll Vector y
            _roll_vec_z = outsim_pack[13] # Roll Vector z
            _pitch_vec_x = outsim_pack[14] # Pitch Vector x
            _pitch_vec_y = outsim_pack[15] # Pitch Vector x
            _pitch_vec_z = outsim_pack[16] # Pitch Vector x
            _pos_sus_rl = outsim_pack[17] # Position of Suspension Rear Left
            _pos_sus_rr = outsim_pack[18] # Position of Suspension Rear Right
            _pos_sus_fl = outsim_pack[19] # Position of Suspension Front Left
            _pos_sus_fr = outsim_pack[20] # Position of Suspension Front Right
            _vel_sus_rl = outsim_pack[21] # Velocity of Suspension Rear Left
            _vel_sus_rr = outsim_pack[22] # Velocity of Suspension Rear Right
            _vel_sus_fl = outsim_pack[23] # Velocity of Suspension Front Left
            _vel_sus_fr = outsim_pack[24] # Velocity of Suspension Front Right
            _vel_wheel_rl = outsim_pack[25] # Velocity of Wheel Rear Left
            _vel_wheel_rr = outsim_pack[26] # Velocity of Wheel Rear Right
            _vel_wheel_fl = outsim_pack[27] # Velocity of Wheel Front Left
            _vel_wheel_fr = outsim_pack[28] # Velocity of Wheel Front Right
            _throttle = outsim_pack[29] # Position Throttle
            _steering = outsim_pack[30] # Position Steer
            _brake = outsim_pack[31] # Position Brake
            _clutch = outsim_pack[32] # Position Clutch
            _gear = outsim_pack[33] # Gear [0 = Neutral, 1 = 1, 2 = 2, ..., 10 = Reverse]
            _gforce_lat = outsim_pack[34] # G-Force Lateral
            _gforce_lon = outsim_pack[35] # G-Force Longitudinal
            _num_lap = outsim_pack[36] # Current Lap
            _eng_rpm = outsim_pack[37] # Speed of Engine [rpm / 10]
            # outsim_pack[38] # ?
            # outsim_pack[39] # ?
            # outsim_pack[40] # ?
            # outsim_pack[41] # ?
            # outsim_pack[42] # ?
            # outsim_pack[43] # ?
            # outsim_pack[44] # ?
            # outsim_pack[45] # ?
            # outsim_pack[46] # ?
            # outsim_pack[47] # ?
            # outsim_pack[48] # ?
            # outsim_pack[49] # ?
            # outsim_pack[50] # ?
            # outsim_pack[51] # Temperature Brake Rear Left ?
            # outsim_pack[52] # Temperature Brake Rear Right ?
            # outsim_pack[53] # Temperature Brake Front Left ?
            # outsim_pack[54] # Temperature Brake Front Right ?
            # outsim_pack[55] # ?
            # outsim_pack[56] # ?
            # outsim_pack[57] # ?
            # outsim_pack[58] # ?
            # outsim_pack[59] # ?
            # outsim_pack[60] # Number of Laps in Total ?
            # outsim_pack[61] # Length of Track in Total
            # outsim_pack[62] # ?
            # outsim_pack[63] # Maximum rpm / 10


            print(f"Velocity: {_vel}")
            print(f"Throttle: {_throttle}")

def main():
    # parser = argparse.ArgumentParser(
    #                 prog='OutSim Data Collect',
    #                 description='Reads OutSim data protocol for vehicle telemetry from Dirt Rally')
    # parser.add_argument('port', default=10001)
    # args = parser.parse_args()

    osdc = OutSim_Data_Collect(10001)


if __name__ == "__main__":
    main()