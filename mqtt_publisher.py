# To start the mosquitto broker, in raspberrypi terminal: mosquitto -c /home/ronnie/Documents/Drive-Guardian/test.config -v

# make sure the test.config file has the correct path.

'''
test.config:
listener 1883
allow_anonymous true
'''

# for mosquitto publisher (windows): cd "C:\Program Files\Mosquitto\" && net start mosquitto
# for mosquitto subscriber: mosquitto_sub -h 192.168.x.x -t test

import paho.mqtt.client as mqtt

broker_address = "192.168.29.216"
client = mqtt.Client(client_id="Publisher")
client.connect(broker_address, 1883, 60)
i = True
while i==True:
    j = input()
    if j == "n":
        # Send "1" to turn the LED on
        client.publish("test", "1")
    else:
        # Send "0" to turn the LED off
        client.publish("test", "0")


client.loop_forever()
