import sys, os
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
import random
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings(action='ignore')


def RandomDate(sdate, edate):

  fmt = "%Y%m%d%H"

  dt_sdate = datetime.strptime(sdate, fmt)  ### str -> datetime
  dt_edate = datetime.strptime(edate, fmt)

  day_list = []

  now = dt_sdate

  while now<=dt_edate:

    ex_sdate = now.strftime(fmt)
    day_list.append(ex_sdate)
    now = now + timedelta(days=1)
  train_list = sorted(random.sample(day_list, int(len(day_list)*8//10)))

  for i in range(len(train_list)):

    day_list.remove(train_list[i])

    valid_list = day_list

  return (train_list, valid_list)


def FileExists(path):

  if not os.path.exists(path):

    print("Can't Find : %s" %(path))

    return False

  else:

    return True


def MakeDataset(options, date_list):

  fmt = "%Y%m%d%H"

  #=== Config
  x_dir  = options.NWPD_dir
  y_dir  = options.ldaps_dir
  var       = options.task
  
  #=== Read Data
  xdata, ydata = [],[]

  for date in date_list:
    print(date)
    xname = "%s/umgl_n128.%s.npz" %(x_dir, date)
    yname = "%s/%s/cat/%s_%s.npy" %(y_dir, var, var, date)

    if not FileExists(xname) or not FileExists(yname):
      print(date, 'is not exist')
      continue
    else:
      xdat = np.load(xname)
      ydat = np.load(yname)
      ydat = ydat[5:-20, 5:]

    #xdat, ydat = ExtrAndResize(fname, nx, ny, upscale_factor, ftime, order)

    if var == "REH":
      xdat = xdat['r'][0,:,:]
      #xdat = np.reshape(xdat, (1, xdat.shape[0], xdat.shape[1]))
      xdat = xdat.swapaxes(0,1)
      xdat = xdat[39:123, 4:]
      xdat = MinMaxscaler(0, 100, xdat)
      #ydat = MinMaxscaler(0, 100, ydat)

    elif var == "T3H":
      xdat = xdat['t'][0,:,:]-273.15
      #xdat = np.reshape(xdat, (1, xdat.shape[0], xdat.shape[1]))
      xdat = xdat.swapaxes(0,1)
      xdat = xdat[39:123, 4:]

      xdat = MinMaxscaler(-50, 50, xdat)
      #ydat = MinMaxscaler(-50, 50, ydat)

    xdata.append(xdat)
    ydata.append(ydat)

  xdata = np.asarray(xdata)
  ydata = np.asarray(ydata)
  return(xdata, ydata)


def MinMaxscaler(Min, Max, data):
  minmax = (data - Min)/(Max - Min)
  return(minmax)


class datasets3d(Dataset):

  def __init__(self, x, y):

    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    self.x = torch.tensor(x, dtype=torch.float)
    self.y = torch.tensor(y, dtype=torch.float)

    if x.shape[0] == y.shape[0]:
      self.rows = x.shape[0]

    else:
      print("x & y nsamples are not matched")
      sys.exit(-1)

  def __len__(self):

    return self.rows

  def __getitem__(self, idx):

    xx = torch.tensor(self.x[idx], dtype=torch.float)
    yy = torch.tensor(self.y[idx], dtype=torch.float)

    return (xx, yy)


def DatasetMaker(args):


  train_list, valid_list, = RandomDate(args.sdate, args.edate)
  trn_fee = int(len(train_list)-(len(train_list)%args.batch_size))
  val_fee = int(len(valid_list)-(len(valid_list)%args.batch_size))
  train_list              = train_list[:trn_fee]
  valid_list              = valid_list[:val_fee]
  print(train_list)
  train_x, train_y        = MakeDataset(args, train_list)
  valid_x, valid_y        = MakeDataset(args, valid_list)

  train_dataset           = datasets3d(train_x, train_y)
  train_loader            = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True, num_workers= 2)

  valid_dataset           = datasets3d(valid_x, valid_y)
  valid_loader            = DataLoader(dataset = valid_dataset, batch_size = args.batch_size, shuffle = False, num_workers= 2)

  return(train_loader, valid_loader)

