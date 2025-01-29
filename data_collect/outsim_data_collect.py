#!/usr/bin/env python3

import os, sys, time, keyboard
import argparse
import socket
import struct
import torch
import datetime
import pandas as pd

timestamp = datetime.datetime.now()
tstamp = timestamp.strftime("%d-%b-%y-%H-%M-%S")
parent_dir = os.path.abspath(os.getcwd())
data_dir = os.path.join(parent_dir,'data')
data_file = os.path.join(data_dir,'OutSim_'+tstamp+'.csv')
if not os.path.exists(data_dir):
    print(f"Creating data directory: {data_dir}")
    os.mkdir(data_dir)

class OutSim_Data_Collect():
    def __init__(self, port):
        self._port = port
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.data_socket.bind(('127.0.0.1',self._port))
        self.run_collect = True

        # Create Lists to store Data
        self._time = [] # Time
        self._lap_time = [] # Time of Current Lap
        self._lap_dist = [] # Distance Driven on Current Lap
        self._overall_dist = [] # Driven Overall
        self._pos_x = [] # Position x
        self._pos_y = [] # Position y
        self._pos_z = [] # Position z
        self._vel = [] # Vehicle Velocity [m/s]
        self._vel_x = [] # Velocity x
        self._vel_y = [] # Velocity y
        self._vel_z = [] # Velocity z
        self._roll_vec_x = [] # Roll Vector x
        self._roll_vec_y = [] # Roll Vector y
        self._roll_vec_z = [] # Roll Vector z
        self._pitch_vec_x = [] # Pitch Vector x
        self._pitch_vec_y = [] # Pitch Vector x
        self._pitch_vec_z = [] # Pitch Vector x
        self._pos_sus_rl = [] # Position of Suspension Rear Left
        self._pos_sus_rr = [] # Position of Suspension Rear Right
        self._pos_sus_fl = [] # Position of Suspension Front Left
        self._pos_sus_fr = [] # Position of Suspension Front Right
        self._vel_sus_rl = [] # Velocity of Suspension Rear Left
        self._vel_sus_rr = [] # Velocity of Suspension Rear Right
        self._vel_sus_fl = [] # Velocity of Suspension Front Left
        self._vel_sus_fr = [] # Velocity of Suspension Front Right
        self._vel_wheel_rl = [] # Velocity of Wheel Rear Left
        self._vel_wheel_rr = [] # Velocity of Wheel Rear Right
        self._vel_wheel_fl = [] # Velocity of Wheel Front Left
        self._vel_wheel_fr = [] # Velocity of Wheel Front Right
        self._throttle = [] # Position Throttle
        self._steering = [] # Position Steer
        self._brake = [] # Position Brake
        self._clutch = [] # Position Clutch
        self._gear = [] # Gear [0 = Neutral, 1 = 1, 2 = 2, ..., 10 = Reverse]
        self._gforce_lat = [] # G-Force Lateral
        self._gforce_lon = [] # G-Force Longitudinal
        self._num_lap = [] # Current Lap
        self._eng_rpm = [] # Speed of Engine [rpm / 10]
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

        self.read_data()
        self.write_data()

    def read_data(self):
        print("Waiting for OutSim Data.")
        while True:
            data = self.data_socket.recv(512) # 64 Float64 values in extradata = 3
            print(f"Data Len: {len(data)}" )

            if(len(data) > 0):
                outsim_pack = struct.unpack('64f', data[0:256])
                # Curtesy of fab1701 https://steamcommunity.com/app/310560/discussions/0/481115363869500839/?l=norwegian
                self._time.append(outsim_pack[0]) # Time
                self._lap_time.append(outsim_pack[1]) # Time of Current Lap
                self._lap_dist.append(outsim_pack[2]) # Distance Driven on Current Lap
                self._overall_dist.append(outsim_pack[3]) # Driven Overall
                self._pos_x.append(outsim_pack[4]) # Position x
                self._pos_y.append(outsim_pack[5]) # Position y
                self._pos_z.append(outsim_pack[6]) # Position z
                self._vel.append(outsim_pack[7]) # Vehicle Velocity [m/s]
                self._vel_x.append(outsim_pack[8]) # Velocity x
                self._vel_y.append(outsim_pack[9]) # Velocity y
                self._vel_z.append(outsim_pack[10]) # Velocity z
                self._roll_vec_x.append(outsim_pack[11]) # Roll Vector x
                self._roll_vec_y.append(outsim_pack[12]) # Roll Vector y
                self._roll_vec_z.append(outsim_pack[13]) # Roll Vector z
                self._pitch_vec_x.append(outsim_pack[14]) # Pitch Vector x
                self._pitch_vec_y.append(outsim_pack[15]) # Pitch Vector x
                self._pitch_vec_z.append(outsim_pack[16]) # Pitch Vector x
                self._pos_sus_rl.append(outsim_pack[17]) # Position of Suspension Rear Left
                self._pos_sus_rr.append(outsim_pack[18]) # Position of Suspension Rear Right
                self._pos_sus_fl.append(outsim_pack[19]) # Position of Suspension Front Left
                self._pos_sus_fr.append(outsim_pack[20]) # Position of Suspension Front Right
                self._vel_sus_rl.append(outsim_pack[21]) # Velocity of Suspension Rear Left
                self._vel_sus_rr.append(outsim_pack[22]) # Velocity of Suspension Rear Right
                self._vel_sus_fl.append(outsim_pack[23]) # Velocity of Suspension Front Left
                self._vel_sus_fr.append(outsim_pack[24]) # Velocity of Suspension Front Right
                self._vel_wheel_rl.append(outsim_pack[25]) # Velocity of Wheel Rear Left
                self._vel_wheel_rr.append(outsim_pack[26]) # Velocity of Wheel Rear Right
                self._vel_wheel_fl.append(outsim_pack[27]) # Velocity of Wheel Front Left
                self._vel_wheel_fr.append(outsim_pack[28]) # Velocity of Wheel Front Right
                self._throttle.append(outsim_pack[29]) # Position Throttle
                self._steering.append(outsim_pack[30]) # Position Steer
                self._brake.append(outsim_pack[31]) # Position Brake
                self._clutch.append(outsim_pack[32]) # Position Clutch
                self._gear.append(outsim_pack[33]) # Gear [0 = Neutral, 1 = 1, 2 = 2, ..., 10 = Reverse]
                self._gforce_lat.append(outsim_pack[34]) # G-Force Lateral
                self._gforce_lon.append(outsim_pack[35]) # G-Force Longitudinal
                self._num_lap.append(outsim_pack[36]) # Current Lap
                self._eng_rpm.append(outsim_pack[37]) # Speed of Engine [rpm / 10]
            else:
                print("No data currently being recieved.")

            if keyboard.is_pressed("q"):
                print(f'Ending Data Collect')
                break
                
    def write_data(self):
        print(f'Writing Data to : {data_file}')
        self.outsim_data = pd.DataFrame({
            'time':self._time,
            'lap_time':self._lap_time,
            'lap_dist': self._lap_dist,
            'overall_dist': self._overall_dist,
            'pos_x': self._pos_x,
            'pos_y': self._pos_y,
            'pos_z': self._pos_z,
            'vel': self._vel,
            'vel_x': self._vel_x,
            'vel_y': self._vel_y,
            'vel_z': self._vel_z,
            'roll_vec_x': self._roll_vec_x,
            'roll_vec_y': self._roll_vec_y,
            'roll_vec_z': self._roll_vec_z,
            'pitch_vec_x': self._pitch_vec_x,
            'pitch_vec_y': self._pitch_vec_y,
            'pitch_vec_z': self._pitch_vec_z,
            'pos_sus_rl': self._pos_sus_rl,
            'pos_sus_rr': self._pos_sus_rr,
            'pos_sus_fl': self._pos_sus_fl,
            'pos_sus_fr': self._pos_sus_fr,
            'vel_sus_rl': self._vel_sus_rl,
            'vel_sus_rr': self._vel_sus_rr,
            'vel_sus_fl': self._vel_sus_fl,
            'vel_sus_fr': self._vel_sus_fr,
            'vel_wheel_rl': self._vel_wheel_rl,
            'vel_wheel_rr': self._vel_wheel_rr,
            'vel_wheel_fl': self._vel_wheel_fl,
            'vel_wheel_fr': self._vel_wheel_fr,
            'throttle': self._throttle,
            'steering': self._steering,
            'brake': self._brake,
            'clutch': self._clutch,
            'gear': self._gear,
            'gforce_lat': self._gforce_lat,
            'gforce_lon': self._gforce_lon,
            'num_lap': self._num_lap,
            'eng_rpm': self._eng_rpm
        })
        self.outsim_data.to_csv(data_file)
        print('Done. Exiting.')
        sys.exit()

def main():
    # parser = argparse.ArgumentParser(
    #                 prog='OutSim Data Collect',
    #                 description='Reads OutSim data protocol for vehicle telemetry from Dirt Rally')
    # parser.add_argument('port', default=10001)
    # args = parser.parse_args()

    osdc = OutSim_Data_Collect(10001)


if __name__ == "__main__":
    main()