'Simple idea around Vehicular Ad Hoc Networks - VANETs'

from random import randint
from threading import Thread 
from mininet.node import Controller, RemoteController, OVSKernelSwitch #
from mininet.link import TCLink #
from mininet.log import setLogLevel, info
from mn_wifi.cli import CLI
from mn_wifi.net import Mininet_wifi
from mn_wifi.link import wmediumd, mesh, adhoc
from mn_wifi.wmediumdConnector import interference



def topology():

    "Create a network."
    net = Mininet_wifi(controller=RemoteController,
                       roads=8,
                       link=wmediumd,
                       wmediumd_mode=interference)

    info("*** Creating nodes\n")
    for id in range(0, 20):
        min_ = randint(1, 5)
        max_ = randint(11, 35)
        net.addCar('car%s' % (id + 1), wlans=2, min_speed=min_, max_speed=max_,range=40)

    rsu11 = net.addAccessPoint('RSU11', ssid='RSU11', mode='g', channel='1',range = 150)
    rsu12 = net.addAccessPoint('RSU12', ssid='RSU12', mode='g', channel='6',range = 150)
    rsu13 = net.addAccessPoint('RSU13', ssid='RSU13', mode='g', channel='11',range = 150)
    rsu14 = net.addAccessPoint('RSU14', ssid='RSU14', mode='g', channel='11',range = 150)
    rsu15 = net.addAccessPoint('RSU15', ssid='RSU15', mode='g', channel='1',range = 150)
    rsu16 = net.addAccessPoint('RSU16', ssid='RSU16', mode='g', channel='6',range = 150)
    rsu17 = net.addAccessPoint('RSU17', ssid='RSU17', mode='g', channel='11',range = 150)
    rsu18 = net.addAccessPoint('RSU18', ssid='RSU18', mode='g', channel='11',range = 150)
    rsu19 = net.addAccessPoint('RSU19', ssid='RSU19', mode='g', channel='1',range = 150)
    
    c1 = net.addController('c1')
    #c1 = net.addController('c1', controller=RemoteController, ip='127.0.0.1', port=6633)
    s1 = net.addSwitch('s1', cls=OVSKernelSwitch)
    
    info("*** Configuring Propagation Model\n")
    net.setPropagationModel(model="logDistance", exp=4)

    info("*** Configuring wifi nodes\n")
    net.configureWifiNodes()

    info("*** Associating and Creating links\n")
    
    net.addLink(rsu11, s1)
    net.addLink(rsu12, s1)
    net.addLink(rsu13, s1)
    net.addLink(rsu14, s1)
    net.addLink(rsu15, s1)
    net.addLink(rsu16, s1)
    net.addLink(rsu17, s1)
    net.addLink(rsu18, s1)
    net.addLink(rsu19, s1)

    net.plotGraph(max_x=1200, max_y=1200)
    
    #c1.plot(position='750,750,0')
    #s1.plot(position='500,500,0')

    
    net.startMobility(time=1)

    info("*** Starting network\n")
    net.build()
    c1.start()  # cho ctl chay
    rsu11.start([c1])
    rsu12.start([c1])
    rsu13.start([c1])
    rsu14.start([c1])
    rsu15.start([c1])
    rsu16.start([c1])
    rsu17.start([c1])
    rsu18.start([c1])
    rsu19.start([c1])
    s1.start([c1])
    info("*** Running CLI\n")
    CLI(net)
    while True:
      for car in net.cars:
               test = 0
               for ap in net.aps:
                 if (car.get_distance_to(ap) <= 150):
                     test = 1
                     car.cmd('iw dev %s-wlan0 set type managed' % car)
                     car.cmd('iw dev %s-wlan0 connect RSU%s' %(car,int(net.aps.index(ap)) + 11))
                     car.setIP('192.168.%s.%s/24' % (int(net.aps.index(ap)) + 1,int(net.cars.index(car)) + 1),
                     intf='%s-wlan0' % car)
                     car.cmd('ip route add 10.10.0.0/24 via 192.168.%s.%s' % (int(net.aps.index(ap)) + 1,int(net.cars.index(car)) + 1))
                     car.cmd('ip route add 192.168.0.0/16 via 192.168.%s.%s' % (int(net.aps.index(ap)) + 1,int(net.cars.index(car)) + 1))
                     car.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')
                     break
               if test == 0:
                     #car.cmd('iwconfig %s-wlan0 mode ad-hoc' % car)
                     car.cmd('iwconfig %s-wlan0 mode ad-hoc essid MyAdHocNetwork channel 1 bitrate 1M txpower 15' % car)
                     car.setIP('10.10.0.%s/24' % (int(net.cars.index(car)) + 1),intf='%s-wlan0' % car)
                     car.cmd('ip route add 192.168.0.0/16 via 10.10.0.%s' % (int(net.cars.index(car)) + 1))
                     car.cmd('ip route add 10.10.0.0/16 via 10.10.0.%s' % (int(net.cars.index(car)) + 1))
                     car.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')
      a = int(input("nhap dau vao: "))
      if a==1:
       CLI(net)
      else:
       break
  
    #CLI(net)
    info("*** Stopping network\n")
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    topology()
 


