original = open('winequality-white.csv', 'r')
new = open('winequality.csv', 'w')

lines = original.readlines()
for line in lines:
    vals = line.split(';')
    line = ','.join(vals)
    new.write(line)

original.close()
new.close()
