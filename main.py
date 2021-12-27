from action_recognition.I3D_kinetics import i3d_kinetics
from action_recognition.TSN3_UCF101 import tsn_ucf101
from action_detection.detect_person import detect

if __name__ == '__main__':

    # Create a ZED camera object
    zed = sl.Camera()

    # Enable recording with the filename
    file_path = "./output/video"
    err = zed.enable_recording(file_path, sl.SVO_COMPRESSION_MODE.H264)
    while !exit_app:
        counter=0
        while counter!=24:
            if zed.grab() == SUCCESS:
                # Each new frame is added to the SVO file
                zed.record()
                counter=+counter
    #detect people in the video frame
    #for each person
        #blur the backgroud and run the following:
    if("tsn_ucf101"):
            output= tsn_ucf101(file_path)
    if("i3d_kinetics400"):
            output= i3d_kinetics('400',file_path)
    if ("i3d_kinetics600"):
            output = i3d_kinetics('600', file_path)
    if ("i3d_kinetics700"):
            output = i3d_kinetics('700', file_path)



            #visualize output on camera




        # Disable recording
        zed.disable_recording()
