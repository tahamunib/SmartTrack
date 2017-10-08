from openalpr import Alpr
import sys
import cv2


alpr = Alpr('us', 'openalpr.conf', 'runtime_data')
if not alpr.is_loaded():
    print("Error loading OpenALPR")
    sys.exit(1)

alpr.set_top_n(20)
alpr.set_default_region("ca")

#img1 = cv2.imread("us-4.jpg",1)
#cv2.imshow('Image',img1)
#cv2.waitKey(0)
results = alpr.recognize_file('us-3.jpg')

i = 0
print results

if len(results['results']) == 0 :
    print "No results found"
    sys.exit()


for plate in results['results']:
    i += 1
    print("Plate #%d" % i)
    print("   %12s %12s" % ("Plate", "Confidence"))
    for candidate in plate['candidates']:
        prefix = "-"
        if candidate['matches_template']:
            prefix = "*"

        print("  %s %12s%12f" % (prefix, candidate['plate'], candidate['confidence']))

# Call when completely done to release memory
alpr.unload()
