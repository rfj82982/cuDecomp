! SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: BSD-3-Clause
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
! 1. Redistributions of source code must retain the above copyright notice, this
!    list of conditions and the following disclaimer.
!
! 2. Redistributions in binary form must reproduce the above copyright notice,
!    this list of conditions and the following disclaimer in the documentation
!    and/or other materials provided with the distribution.
!
! 3. Neither the name of the copyright holder nor the names of its
!    contributors may be used to endorse or promote products derived from
!    this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#define CHECK_CUDECOMP_EXIT(f) if (f /= CUDECOMP_RESULT_SUCCESS) call exit(1)

! Test forward and backward FFT to retrieve the input result

program main
  use cudafor
  use cudecomp
  use cufft
  use mpi

  implicit none

  ! Command line arguments
  ! grid dimensions
  integer :: nx, ny, nz
  integer :: comm_backend
  integer :: pr, pc

  complex(8), allocatable :: phi(:), ua(:,:,:)
  complex(8), device, allocatable :: phi_d(:)
  complex(8), pointer, device, contiguous :: work_d(:)

  ! MPI
  integer :: rank, ranks, ierr
  integer :: localRank, localComm

  ! cudecomp
  type(cudecompHandle) :: handle
  type(cudecompGridDesc) :: grid_desc
  type(cudecompGridDescConfig) :: config
  type(cudecompGridDescAutotuneOptions) :: options

  integer :: pdims(2)
  integer :: gdims(3)
  integer :: npx, npy, npz
  type(cudecompPencilInfo) :: piX, piY, piZ
  integer(8) :: nElemX, nElemY, nElemZ, nElemWork

  ! CUFFT
  integer :: planX_F, planY_F, planZ_F
  integer :: planX_B, planY_B, planZ_B
  integer :: batchsize
  integer :: status

  logical :: skip_next
  character(len=16) :: arg
  integer :: i, j, k

  ! Test
  integer :: itest, ntest=10
  real(8) :: t1, t2, t3, t4

  ! MPI initialization

  call mpi_init(ierr)
  if (ierr /= MPI_SUCCESS) write(*,*) 'mpi_init failed: ', ierr
  call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)
  if (ierr /= MPI_SUCCESS) write(*,*) 'mpi_comm_rank failed: ', ierr
  call mpi_comm_size(MPI_COMM_WORLD, ranks, ierr)
  if (ierr /= MPI_SUCCESS) write(*,*) 'mpi_comm_size failed: ', ierr

  call mpi_comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, localComm, ierr)
  if (ierr /= MPI_SUCCESS) write(*,*) 'mpi_comm_split_type failed: ', ierr
  call mpi_comm_rank(localComm, localRank, ierr)
  if (ierr /= MPI_SUCCESS) write(*,*) 'mpi_comm_rank on local rank failed: ', ierr
  ierr = cudaSetDevice(localRank)

  ! Parse command-line arguments
  nx = 64
  ny = 64
  nz = 64
  pr = 0
  pc = 0
  comm_backend = CUDECOMP_TRANSPOSE_COMM_NCCL

  skip_next = .false.
  do i = 1, command_argument_count()
    if (skip_next) then
      skip_next = .false.
      cycle
    end if
    call get_command_argument(i, arg)
    select case(arg)
      case('--nx')
        call get_command_argument(i+1, arg)
        read(arg, *) nx
        skip_next = .true.
      case('--ny')
        call get_command_argument(i+1, arg)
        read(arg, *) ny
        skip_next = .true.
      case('--nz')
        call get_command_argument(i+1, arg)
        read(arg, *) nz
        skip_next = .true.
      case('--backend')
        call get_command_argument(i+1, arg)
        read(arg, *) comm_backend
        skip_next = .true.
      case('--pr')
        call get_command_argument(i+1, arg)
        read(arg, *) pr
        skip_next = .true.
      case('--pc')
        call get_command_argument(i+1, arg)
        read(arg, *) pc
        skip_next = .true.
      case default
        print*, "Unknown argument."
        call exit(1)
    end select
  end do

  ! check for valid Mx/My/Mz
  if (nx==0 .or. ny==0 .or. nz==0) then
     if (rank == 0) write(*,*) 'Mx/My/Mz is 0 thus solution is u=0'
     call MPI_Finalize(ierr)
     stop
  else if (rank == 0) then
     write(*,"('Running on ', i0, ' x ', i0, ' x ', i0, ' spatial grid ...')") &
          nx, ny, nz
  end if

  ! cudecomp initialization

  CHECK_CUDECOMP_EXIT(cudecompInit(handle, MPI_COMM_WORLD))

  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(config))
  pdims = [pr, pc]
  config%pdims = pdims
  gdims = [nx, ny, nz]
  config%gdims = gdims
  config%transpose_comm_backend = comm_backend
  config%transpose_axis_contiguous = .true.

  CHECK_CUDECOMP_EXIT(cudecompGridDescAutotuneOptionsSetDefaults(options))
  options%dtype = CUDECOMP_DOUBLE_COMPLEX
  if (comm_backend == 0) then
    options%autotune_transpose_backend = .true.
  endif

  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, grid_desc, config, options))

  if (rank == 0) then
     write(*,"('Running on ', i0, ' x ', i0, ' process grid ...')") config%pdims(1), config%pdims(2)
     write(*,"('Using ', a, ' backend ...')") cudecompTransposeCommBackendToString(config%transpose_comm_backend)
  end if

  ! get pencil info
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, piX, 1))
  nElemX = piX%size
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, piY, 2))
  nElemY = piY%size
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, piZ, 3))
  nElemZ = piZ%size

  ! get workspace size
  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc, nElemWork))

  ! CUFFT initialization

  batchSize = piX%shape(2)*piX%shape(3)
  status = cufftPlan1D(planX_F, nx, CUFFT_Z2Z, batchSize)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating X plan'

  batchSize = piY%shape(2)*piY%shape(3)
  status = cufftPlan1D(planY_F, ny, CUFFT_Z2Z, batchSize)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating Y plan'

  batchSize = piZ%shape(2)*piZ%shape(3)
  status = cufftPlan1D(planZ_F, nz, CUFFT_Z2Z, batchSize)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating Z plan'

  batchSize = piX%shape(2)*piX%shape(3)
  status = cufftPlan1D(planX_B, nx, CUFFT_Z2Z, batchSize)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating X plan'

  batchSize = piY%shape(2)*piY%shape(3)
  status = cufftPlan1D(planY_B, ny, CUFFT_Z2Z, batchSize)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating Y plan'

  batchSize = piZ%shape(2)*piZ%shape(3)
  status = cufftPlan1D(planZ_B, nz, CUFFT_Z2Z, batchSize)
  if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating Z plan'

  ! allocate arrays

  allocate(phi(max(nElemX, nElemY, nElemZ)))
  allocate(phi_d, mold=phi)
  allocate(ua(nx, piX%shape(2), piX%shape(3)))
  CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc, work_d, nElemWork))

  ! initialize phi and analytical solution
  block
    complex(8), pointer :: phi3(:,:,:)
    real(8) :: dr
    integer :: jl, kl, jg, kg
    npx = piX%shape(1)
    npy = piX%shape(2)
    npz = piX%shape(3)
    call c_f_pointer(c_loc(phi), phi3, [npx, npy, npz])

    do kl = 1, npz
       kg = piX%lo(3) + kl - 1
       do jl = 1, npy
          jg = piX%lo(2) + jl - 1
          do i = 1, nx
             dr = (real(i)/real(nx))*(real(jg)/real(ny))*(real(kg)/real(nz))
             phi3(i,jl,kl) = cmplx(dr,dr)
             ua(i,jl,kl)   = cmplx(dr,dr) 
          enddo
       enddo
    enddo
  end block

  ! H2D transfer
  phi_d = phi

  t2 = 0.d0
  t4 = 0.d0

  do itest=1,ntest
     t1 = MPI_WTIME()
     ! phi(x,y,z) -> phi(kx,y,z)
     status = cufftExecZ2Z(planX_F, phi_d, phi_d, CUFFT_FORWARD)
     if (status /= CUFFT_SUCCESS) write(*,*) 'X forward error: ', status
     ! phi(kx,y,z) -> phi(y,z,kx)
     CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(handle, grid_desc, phi_d, phi_d, work_d, CUDECOMP_DOUBLE_COMPLEX))
     ! phi(y,z,kx) -> phi(ky,z,kx)
     status = cufftExecZ2Z(planY_F, phi_d, phi_d, CUFFT_FORWARD)
     if (status /= CUFFT_SUCCESS) write(*,*) 'Y forward error: ', status
     ! phi(ky,z,kx) -> phi(z,kx,ky)
     CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(handle, grid_desc, phi_d, phi_d, work_d, CUDECOMP_DOUBLE_COMPLEX))
     ! phi(z,kx,ky) -> phi(kz,kx,ky)
     status = cufftExecZ2Z(planZ_F, phi_d, phi_d, CUFFT_FORWARD)
     if (status /= CUFFT_SUCCESS) write(*,*) 'Z forward error: ', status
     t2 = t2 + MPI_WTIME()-t1

     t3 = MPI_WTIME()
     ! phi(kz,kx,ky) -> phi(z,kx,ky)
     status = cufftExecZ2Z(planZ_B, phi_d, phi_d, CUFFT_INVERSE)
     if (status /= CUFFT_SUCCESS) write(*,*) 'Z inverse error: ', status
     ! phi(z,kx,ky) -> phi(ky,z,kx)
     CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(handle, grid_desc, phi_d, phi_d, work_d, CUDECOMP_DOUBLE_COMPLEX))
     ! phi(ky,z,kx) -> phi(y,z,kx)
     status = cufftExecZ2Z(planY_B, phi_d, phi_d, CUFFT_INVERSE)
     if (status /= CUFFT_SUCCESS) write(*,*) 'Y inverse error: ', status
     ! phi(y,z,kx) -> phi(kx,y,z)
     CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(handle, grid_desc, phi_d, phi_d, work_d, CUDECOMP_DOUBLE_COMPLEX))
     ! phi(kx,y,z) -> phi(x,y,z)
     status = cufftExecZ2Z(planX_B, phi_d, phi_d, CUFFT_INVERSE)
     if (status /= CUFFT_SUCCESS) write(*,*) 'X inverse error: ', status
     t4 = t4 + MPI_WTIME()-t3
     
     ! normalisation
     block
       complex(8), device, pointer :: phi3(:,:,:)
       integer :: jl, kl, jg, kg
       npx = piX%shape(1)
       npy = piX%shape(2)
       npz = piX%shape(3)
       call c_f_pointer(c_devloc(phi_d), phi3, [npx, npy, npz])

       !$cuf kernel do (2)
       do kl = 1, npz
          kg = piX%lo(3) + kl - 1
          do jl = 1, npy
             jg = piX%lo(2) + jl - 1
             do i = 1, nx
                phi3(i,jl,kl) = phi3(i,jl,kl)/(nx*ny*nz)
             enddo
          enddo
       enddo
     end block
  enddo

  call MPI_ALLREDUCE(t2, t1, 1, MPI_DOUBLE_PRECISION, MPI_SUM, &
                      MPI_COMM_WORLD, ierr)
  t1 = t1 / real(ranks)
  call MPI_ALLREDUCE(t4, t3, 1, MPI_DOUBLE_PRECISION, MPI_SUM, &
                      MPI_COMM_WORLD, ierr)
  t3 = t3 / real(ranks)
  t4 = t1+t3
  if (rank == 0) then
    write(*,*) 'Time Forward ', t1
    write(*,*) 'Time Backwards ', t3
    write(*,*) 'Time TOT  ', t4
  endif


  ! H2D transfer

  phi = phi_d

  ! check results

  block
    complex(8), pointer :: phi3(:,:,:)
    real(8) :: err, maxErr = -1.0
    integer :: jl, kl
    npx = piX%shape(1)
    npy = piX%shape(2)
    npz = piX%shape(3)
    call c_f_pointer(c_loc(phi), phi3, [npx, npy, npz])


    do kl = 1, npz
       do jl = 1, npy
          do i = 1, nx
             err = abs(ua(i,jl,kl)-phi3(i,jl,kl))
             if (err > maxErr) maxErr = err
          enddo
       enddo
    enddo

    write(*,"('[', i0, '] Max Error: ', e12.6)") rank, maxErr
  end block

  ! cleanup

  status = cufftDestroy(planX_F)
  if (status /= CUFFT_SUCCESS) write(*,*) 'X plan destroy: ', status
  status = cufftDestroy(planY_F)
  if (status /= CUFFT_SUCCESS) write(*,*) 'Y plan destroy: ', status
  status = cufftDestroy(planZ_F)
  if (status /= CUFFT_SUCCESS) write(*,*) 'Z plan destroy: ', status
  status = cufftDestroy(planX_B)
  if (status /= CUFFT_SUCCESS) write(*,*) 'X plan destroy: ', status
  status = cufftDestroy(planY_B)
  if (status /= CUFFT_SUCCESS) write(*,*) 'Y plan destroy: ', status
  status = cufftDestroy(planZ_B)
  if (status /= CUFFT_SUCCESS) write(*,*) 'Z plan destroy: ', status

  deallocate(phi, phi_d, ua)

  CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, work_d))
  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc))
  CHECK_CUDECOMP_EXIT(cudecompFinalize(handle))

  call mpi_finalize(ierr)

end program main
