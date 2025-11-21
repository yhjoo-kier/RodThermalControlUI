class PIDController:
    def __init__(self, kp=1.0, ki=0.1, kd=0.01, target_temp=50.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_temp = target_temp
        self.integral = 0.0
        self.prev_error = 0.0
        
    def predict(self, current_temp_array):
        # Control based on mean temperature
        mean_temp = sum(current_temp_array) / len(current_temp_array)
        error = self.target_temp - mean_temp
        
        self.integral += error
        derivative = error - self.prev_error
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        
        return max(0.0, min(50.0, output))
