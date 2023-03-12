from source.python.runtime import get_system_info

if __name__ == '__main__' :
	print()

	for key, value in get_system_info().items() :
		print('{:16s} : {}'.format(key, value))

	print()
