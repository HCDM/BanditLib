import sys
from sets import Set
import random

user_arm_tag = []

#remove duplicate events
fin = open(sys.argv[1], 'r')
fin.readline()
last = {}
for line in fin:	
	arr = line.strip().split('\t')
	t = {}
	t['uid'] = int(arr[0])
	t['aid'] = int(arr[1])	
	t['tstamp'] = int(arr[3])
	#print t['tstamp']
	if not t == last:
		last = t
		user_arm_tag.append(t)
print 'event number: '+str(len(user_arm_tag))

#filter arm pool for each user
user_arm_pool = {}
arm_pool = Set([])
user_pool = Set([])
for t in user_arm_tag:
	arm_pool.add(t['aid'])
	user_pool.add(t['uid'])
arm_shuffle = list(arm_pool)
random.shuffle(arm_shuffle)
arm_train_test_split = 0.5
arm_pool_train = Set([])
arm_pool_test = Set([])
for arm in arm_shuffle[:int(len(arm_pool)*arm_train_test_split)]:
	arm_pool_train.add(arm)
for arm in arm_shuffle[int(len(arm_pool)*arm_train_test_split):]:
	arm_pool_test.add(arm)

user_shuffle = list(user_pool)
random.shuffle(user_shuffle)
user_train_test_split = 0.5
user_pool_train = Set([])
user_pool_test = Set([])
for user in user_shuffle[:int(len(user_pool)*user_train_test_split)]:
	user_pool_train.add(user)
for user in user_shuffle[int(len(user_pool)*user_train_test_split):]:
	user_pool_test.add(user)

for t in user_arm_tag:
	if not (t['uid'] in user_arm_pool):
		user_arm_pool[t['uid']] = arm_pool.copy()		
	if t['aid'] in user_arm_pool[t['uid']]:
		user_arm_pool[t['uid']].remove(t['aid'])	
random.shuffle(user_arm_tag)

#generate random arm_pool and write to file
fout_Part0 = open(sys.argv[1].split('/')[0]+'/processed_events_shuffled_MFCollab_part0.dat','w') # upper left
fout_Part0.write('userid	timestamp	arm_pool\n')
fout_Part1 = open(sys.argv[1].split('/')[0]+'/processed_events_shuffled_MFCollab_part1.dat','w') # three part
fout_Part1.write('userid	timestamp	arm_pool\n')
fout_Part2 = open(sys.argv[1].split('/')[0]+'/processed_events_shuffled_MFCollab_part2.dat','w') # upper right
fout_Part2.write('userid	timestamp	arm_pool\n')
for t in user_arm_tag:	
	if t['uid'] in user_pool_train and t['aid'] in arm_pool_train:
		random_pool_0 = [t['aid']]+random.sample(user_arm_pool[t['uid']]-arm_pool_test, 24)
		#print random_pool_1
		fout_Part0.write(str(t['uid'])+'\t'+str(t['tstamp'])+'\t'+str(random_pool_0)+'\n')	
		fout_Part1.write(str(t['uid'])+'\t'+str(t['tstamp'])+'\t'+str(random_pool_0)+'\n')	
	elif t['uid'] in user_pool_test or t['aid'] in arm_pool_train:
		#print t['aid']
		random_pool_1 = [t['aid']]+random.sample(user_arm_pool[t['uid']]-arm_pool_test, 24)
		#print random_pool_1
		fout_Part1.write(str(t['uid'])+'\t'+str(t['tstamp'])+'\t'+str(random_pool_1)+'\n')		
	if t['uid'] in user_pool_train and t['aid'] in arm_pool_test:
		random_pool_2 = [t['aid']]+random.sample(user_arm_pool[t['uid']]-arm_pool_train, 24)
		fout_Part2.write(str(t['uid'])+'\t'+str(t['tstamp'])+'\t'+str(random_pool_2)+'\n')
fout_Part0.close()
fout_Part1.close()
fout_Part2.close()