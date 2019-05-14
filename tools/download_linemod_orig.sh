cd ~
mkdir LINEMOD_ORIG
cd LINEMOD_ORIG

# http://campar.in.tum.de/Main/StefanHinterstoisser

wget -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/ape.zip'
wget -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/benchviseblue.zip'
wget -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/bowl.zip'
wget -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/can.zip'
wget -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/cat.zip'
wget -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/cup.zip'
wget -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/driller.zip'
wget -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/duck.zip'
wget -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/glue.zip'
wget -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/holepuncher.zip'
wget -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/iron.zip'
wget -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/lamp.zip'
wget -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/phone.zip'
wget -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/cam.zip'
wget -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/eggbox.zip'

for file in `ls`; do unzip $file; done;

