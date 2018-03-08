f1=open('reshape.txt','w')
a=0
b=[]
for i in open('data.txt','r'):
	i=i.split()
	# b.append(i)
	a=a+1

	if a == 3:
		f1.write(i[0])
		f1.write(',')
		f1.write('\n')
		a=0
	else:
		f1.write(i[0])
		f1.write(',')

