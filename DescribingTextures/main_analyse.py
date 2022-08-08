from DomainToText_AMLProject.eval_ensamble import analyse, analyse2

print('Bargemon')
targets = ['ArtPainting','Cartoon','Photo','Sketch']
checkpath_best = 'results/res_hard_W0515_margin15/triplet_match/temp/checkpoints/BEST_checkpoint.pth'
checkpath_last = 'results/res_hard_W0515_margin15/triplet_match/temp/checkpoints/LAST_checkpoint.pth'

results = {}
for target in targets:
    print("TARGET ",target)
    outp = analyse(checkpath_best, target)
    results[target+'BEST'] = outp
    print("2")
    outp = analyse(checkpath_last, target)
    results[target+'LAST'] = outp


for target in targets:
    print("TARGET ",target)
    outp = analyse2(checkpath_best, target)
    results[target+'BEST'+'-v2'] = outp
    print("2")
    outp = analyse2(checkpath_last, target)
    results[target+'LAST'+'-v2'] = outp

print(results)


