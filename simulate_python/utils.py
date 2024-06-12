import numpy as np
DTYPE = np.float32

# parameters: filename and simulation state mj_data
# takes sensordata vector from mj_data and appends it to filename csv
def log(filename, mj_data):
    datacopy = mj_data.sensordata.copy()
    datacopy.resize(datacopy.shape[0], 1)
    datacopy = datacopy.T

    # print(datacopy.shape)
    # mujoco.mju_writeLog('a', np.array2string(mj_data.sensordata))
    # np.savetxt("testdata.csv", datacopy.T, delimiter=',')

    with open(filename,'a') as fd:
        np.savetxt(fd, datacopy, delimiter=',')
        # writer = csv.writer(fd, delimiter=';')
        # writer.writerow(datacopy)
        fd.close()

    # prints foot forces to console
    # print("FL: ", end = " ")
    # print(mj_data.sensordata[52], end = " ")

    # print("FR: ", end = " ")
    # print(mj_data.sensordata[53], end = " ")

    # print("RL: ", end = " ")
    # print(mj_data.sensordata[54], end = " ")

    # print("RR: ", end = " ")
    # print(mj_data.sensordata[55])

# parameters: filename
# clears data in csv file and inserts header labels
def initialize(filename):
    # clears data in file
    f = open(filename, "w+")
    f.close()

    leg = np.array(["FR_", "FL_", "RR_", "RL_"])
    link = np.array(["hip", "thigh", "calf"])

    header = np.empty((1,0), dtype=np.dtype(str))

    # joint pos
    for i in leg:
        for j in link:
            sensorname = i + j + "_q"
            header = np.append(header, np.array([sensorname]))

    # joint vel
    for i in leg:
        for j in link:
            sensorname = i + j + "_dq"
            header = np.append(header, np.array([sensorname]))

    # joint torque
    for i in leg:
        for j in link:
            sensorname = i + j + "_tau"
            header = np.append(header, np.array([sensorname]))

    # quaternions
    # framepos and framelinvel
    quat_and_frame_sensors = ["imu_quat_1", "imu_quat_2", "imu_quat_3", "imu_quat_4", 
                    "imu_gyro_1", "imu_gyro_2", "imu_gyro_3", 
                    "imu_acc_1", "imu_acc_2", "imu_acc_3", 
                    "imu_pos_1", "imu_pos_2", "imu_pos_3", 
                    "imu_vel_1", "imu_vel_2", "imu_vel_3"]
    header = np.append(header, quat_and_frame_sensors)

    # foot force
    for i in leg:
        sensorname = i + "foot_force"
        header = np.append(header, np.array([sensorname]))

    # print(header)

    # print to csv
    with open(filename,'a') as fd:
        header_str = ','.join(header)  # Join header elements with commas
        fd.write(header_str + '\n')
        fd.close()


class Quaternion:
    def __init__(self, w:float=1, x:float=0, y:float=0, z:float=0):
        self.w = DTYPE(w)
        self.x = DTYPE(x)
        self.y = DTYPE(y)
        self.z = DTYPE(z)
        self._norm = np.sqrt(self.w*self.w+self.x*self.x+self.y*self.y+self.z*self.z, dtype=DTYPE)

    def toNumpy(self):
        """convert to an (4,1) numpy array"""
        return np.array([self.w,self.x,self.y,self.z], dtype=DTYPE).reshape((4,1))
    
    def unit(self):
        """return the unit quaternion"""
        return Quaternion(self.w/self._norm,self.x/self._norm,self.y/self._norm,self.z/self._norm)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def reverse(self):
        """return the reverse rotation representation as the same as the transpose op of rotation matrix"""
        return Quaternion(-self.w,self.x,self.y,self.z)

    def inverse(self):
        return Quaternion(self.w/(self._norm*self._norm),-self.x/(self._norm*self._norm),-self.y/(self._norm*self._norm),-self.z/(self._norm*self._norm))
    
    def __str__(self) -> str:
        return '['+str(self.w)+', '+str(self.x)+', '+str(self.y)+', '+str(self.z)+']'


def quat_to_rpy(q:Quaternion) -> np.ndarray:
    """
    Convert a quaternion to RPY. Return
    angles in (roll, pitch, yaw).
    """
    rpy = np.zeros((3,1), dtype=DTYPE)
    as_ = np.min([-2.*(q.x*q.z-q.w*q.y),.99999])
    # roll
    rpy[0] = np.arctan2(2.*(q.y*q.z+q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z)
    # pitch
    rpy[1] = np.arcsin(as_)
    # yaw
    rpy[2] = np.arctan2(2.*(q.x*q.y+q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)
    return rpy

