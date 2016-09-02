// This file is part of primal-dual-toolbox.
//
// Copyright (C) 2018 Kerstin Hammernik <hammernik at icg dot tugraz dot at>
// Institute of Computer Graphics and Vision, Graz University of Technology
// https://www.tugraz.at/institute/icg/research/team-pock/
//
// primal-dual-toolbox is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// primal-dual-toolbox is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <map>

#include "primaldualtoolbox_api.h"

#include <iu/iumath/typetraits.h>
#include <iu/iucore.h>

/** \brief Template argument-free interface wrapper for data items of type iu::LinearDeviceMemory.
 *
 * This class is only a wrapper class to be able to store iu::LinearDeviceMemory of different
 * data types and dimensions in the same vector. This class is only used for data handling in
 * the OperatorBase class! Derived classes do not have to no anything about it!
 */
class PRIMALDUALTOOLBOX_DLLAPI IDataItem
{
public:
  /** Constructor. */
  IDataItem()
  {
  }

  /** Destructor. */
  virtual ~IDataItem()
  {
  }
};

typedef std::map<std::string, std::string> OpConfigDict;

/** OpConfig holds the configuration parameters for the Operator.
 */
class PRIMALDUALTOOLBOX_DLLAPI OpConfig
{
public:
  /** Special constructor taking an OpConfigDict, defining the configuration parameters for the operator, as input.*/
  OpConfig(const OpConfigDict &config) :
      dict_(config)
  {
  }

  /** Constructor */
  OpConfig()
  {
  }

  /** Get the value for a specific configuration parameter.
   * \param key configuration parameter
   * \return configuration value
   */
  template<typename out>
  out getValue(const std::string &key) const
  {
    auto iter = dict_.find(key);
    if (iter == dict_.end())
    {
      // could not find the key
      throw IuException("key net found!", __FILE__, __FUNCTION__, __LINE__);
    }
    else
    {
      std::stringstream ss;
      ss << iter->second;
      out ret;
      ss >> ret;
      if (ss.fail())
        throw IuException("could not parse!", __FILE__, __FUNCTION__, __LINE__);
      return ret;
    }
  }

  /** Get the string representation for a specific configuration parameter.
   * \param key configuration parameter
   * \return configuration value as string
   */
  std::string getStr(const std::string &key) const
  {
    auto iter = dict_.find(key);
    if (iter == dict_.end())
    {
      // could not find the key
      throw IuException("key net found!", __FILE__, __FUNCTION__, __LINE__);
    }
    else
    {
      return iter->second;
    }
  }

  /** Check if the dictionary has a specific configuration parameter.
   * \param key configuration parameter
   * \return true if key is in the dictionary.
   */
  bool hasKey(const std::string &key) const
  {
    auto iter = dict_.find(key);
    if (iter == dict_.end())
    {
      return false;
    }
    else
    {
      return true;
    }
  }

  /** Get size of the dictionary
   */
  int size() const
  {
    return dict_.size();
  }

  /** Overload operator<< for pretty printing.
   */
  friend std::ostream& operator<<(std::ostream & out, OpConfig const& conf)
  {
    int i = 0;
    for (auto iter = conf.dict_.begin(); iter != conf.dict_.end(); ++iter)
      out << iter->first << ":" << iter->second
          << ((++i < conf.dict_.size()) ? "," : "");
    return out;
  }

private:
  /** Dictionary */
  OpConfigDict dict_;
};

/** \brief Wrapper for data items of type iu::LinearDeviceMemory.
 *
 * This class is a wrapper class to be able to store iu::LinearDeviceMemory of different
 * data types and dimensions in the same vector having the same template arguments as the
 * linear device memory. This class is only used for data handling in the OperatorBase
 * class! Derived classes do not have to know anything about it!
 */
template<typename PixelType, unsigned int Ndim>
class DataItem : public IDataItem
{
public:
  /** Special Constructor.
   *
   * A DataItem is initialized with a LinearHostMemory. It is responsible for data handling
   * of the LinearDeviceMemory, thus copies the content from host to device.
   * @param: hostmem iu::LinearHostMemory
   */
  DataItem(const iu::LinearHostMemory<PixelType, Ndim> &hostmem)
  {
    devicemem_.reset(
        new iu::LinearDeviceMemory<PixelType, Ndim>(hostmem.size()));
    iu::copy(&hostmem, devicemem_.get());
  }

  /** Destructor */
  virtual ~DataItem()
  {
  }

  /** Returns the real pointer to the iu::LinearDeviceMemory
   * @return real pointer to the iu::LinearDeviceMemory
   */
  iu::LinearDeviceMemory<PixelType, Ndim>* data()
  {
    return devicemem_.get();
  }
private:
  /** Unique pointer of the iu:::LinearDeviceMemory */
  std::unique_ptr<iu::LinearDeviceMemory<PixelType, Ndim> > devicemem_;
};

/** \brief Abstract OperatorBase class that implements the general interface of operator.
 *
 * The abstract OperatorBase class is defined via an InputType and OutputType. It can
 * perform forward and adjoint operations. If additional constants are required,
 * they can be simply set via addConstant() and do not rely on any specific pixel
 * type and dimension! It handles the constants using the IDataItem and DataItem classes
 * that allow to store data of different dimensions and pixel types in the same std::vector.
 * The constants can be simply extracted based on their pixel types and dimensions
 * by calling getConstant. Additionally, basic checks are performed.
 */
template<typename InputType, typename OutputType>
class PRIMALDUALTOOLBOX_DLLAPI OperatorBase
{
public:
  /** Special constructor.
   * @param num_required_constants Number of additionally required constants
   * @param name Name of the operator
   */
  OperatorBase(unsigned int num_required_constants, const std::string & name,
               const OpConfig &config = OpConfig(), int min_num_required_config_params=0,
               int max_num_required_config_params=0) :
      num_required_constants_(num_required_constants), name_(name), config_(config),
      min_num_required_config_params_(min_num_required_config_params),
      max_num_required_config_params_(max_num_required_config_params)
  {
    if (config_.size() < min_num_required_config_params_ || config.size() > max_num_required_config_params_)
    {
      // too few config parameters
      std::stringstream ss;
      ss << "Operator '" << name_ << "': ";
      ss << "Invalid amount of configuration parameters (min: "
          << min_num_required_config_params_ << " max: " << max_num_required_config_params_ << ") "
              "required, got (" << config.size() << ").";
      throw IuException(ss.str(), __FILE__, __FUNCTION__, __LINE__);
    }
  }

  /** Destructor */
  virtual ~OperatorBase()
  {
  }

  /** Add a constant. Copy the content from host to device.
   * @param hostmem iu::LinearHostMemory */
  template<typename PixelType, unsigned int Ndim>
  void addConstant(const iu::LinearHostMemory<PixelType, Ndim> &hostmem)
  {
    // Check if the number of required constants is not exceeded yet.
    if (constants_.size() < num_required_constants_)
    {
      constants_.push_back(
          std::unique_ptr<DataItem<PixelType, Ndim>>(
              new DataItem<PixelType, Ndim>(hostmem)));

    }
    else
    {
      // throw exception
      std::stringstream ss;
      ss << "Operator '" << name_ << "': ";
      ss << "Too many InputTypeConstants. Could not add new constant.'";
      throw IuException(ss.str(), __FILE__, __FUNCTION__, __LINE__);
    }
  }

  /** Get a constant based on their PixelType and Ndim. In the derived class,
   * this can be called using this->template getConstant<PixelType, Ndim>(idx).
   * @param idx Index of the constant
   * @return iu::LinearDeviceMemory */
  template<typename PixelType, unsigned int Ndim>
  iu::LinearDeviceMemory<PixelType, Ndim>* getConstant(unsigned int idx)
  {
    // Check if index is valid
    if (idx >= constants_.size())
    {
      std::stringstream ss;
      ss << "Operator '" << name_ << "': ";
      ss << "Requested index (" << idx
          << ") is not valid. Constants available: " << constants_.size();
      throw IuException(ss.str(), __FILE__, __FUNCTION__, __LINE__);
    }

    auto item = dynamic_cast<DataItem<PixelType, Ndim> *>(constants_[idx].get());

    // Check if item could be casted to right PixelType and Ndim
    if (item == NULL)
    {
      std::stringstream ss;
      ss << "Operator '" << name_ << "': ";
      ss << "Data at idx (" << idx
          << ") cannot be cast to type iu::LinearDeviceMemory<PixelType, Ndim> with PixelType="
          << iu::type_trait < PixelType > ::name() << " Ndim=" << Ndim;
      throw IuException(ss.str(), __FILE__, __FUNCTION__, __LINE__);
    }

    return item->data();
  }

  /** Perform forward operation on src and store it in dst. Additionally make checks
   * on operator.
   * @param[in] src Source image of type InputType
   * @param[out] dst Destination image of type OutputType
   */
  void forward(const InputType & src, OutputType & dst)
  {
    operatorCheck(src, dst);
    executeForward(src, dst);
  }

  /** Perform adjoint operation on src and store it in dst. Additionally make checks
   * on operator.
   * @param[in] src Source image of type OutputType
   * @param[out] dst Destination image of type InputType
   */
  void adjoint(const OutputType & src, InputType & dst)
  {
    operatorCheck(dst, src);
    executeAdjoint(src, dst);
  }

  /** Pure virtual forward method that has to be implemented in derived class. This method
   * is called in forward.
   * @param[in] src Source image of type InputType
   * @param[out] dst Destination image of type OutputType
   */
  virtual void executeForward(const InputType & src, OutputType & dst) = 0;

  /** Pure virtual adjoint method that has to be implemented in derived class. This method
   * is called in adjoint.
   * @param[in] src Source image of type OutputType
   * @param[out] dst Destination image of type InputType
   */
  virtual void executeAdjoint(const OutputType & src, InputType & dst) = 0;

  /** Pure virtual method that has to be implemented in derived class.
   *  Get the input size from given output.
   * @param[in] in Source image of type OutputType
   */
  virtual iu::Size<InputType::ndim> getInputSize(const OutputType & out) = 0;

  /** Pure virtual method that has to be implemented in derived class.
   *  Get the output size from given input.
   * @param[in] in Source image of type InputType
   */
  virtual iu::Size<OutputType::ndim> getOutputSize(const InputType & in) = 0;

  /** Print information about the operator. */
  virtual std::string print() const
  {
    std::stringstream msg;
    msg << name_ << ":";
    msg << " Added constants: (" << constants_.size() << "/"
        << num_required_constants_ << ")";
    return msg.str();
  }

  /** Overload operator << for printing operator information. */
  friend std::ostream& operator<<(std::ostream& msg,
                                  const OperatorBase<InputType, OutputType>& op)
  {
    return msg << op.print();
  }

  /** No copies are allowed. */
  OperatorBase(OperatorBase const&) = delete;

  /** No assignments are allowed. */
  void operator=(OperatorBase const&) = delete;

protected:
  /** Operator configuration */
  OpConfig config_;

  /** Pure virtual size check method that has to be implemented in derived class. This method
   * is called in operatorCheck.
   * @param[in] in Source image of type InputType
   * @param[in] out Destination image of type OutputType
   */
  virtual void sizeCheck(const InputType & in, const OutputType & out) = 0;

  /** Pure virtual method to check the operator's adjointness. This has to be implemented in derived class.
   */
  virtual void adjointnessCheck() = 0;

  /** Get config parameter as double.
   * \param id Identifier for config parameter
   */
  double getConfigDouble(const std::string &id) const
  {
    try
    {
      return config_.getValue<double>(id);
    }
    catch (std::exception &e)
    {
      // could not find the config parameter
      std::stringstream ss;
      ss << e.what() << std::endl;
      ss << "Operator '" << name_ << "': ";
      ss << "could not parse configuration parameter '" << id << "'!"
          << std::endl;
      throw IuException(ss.str(), __FILE__, __FUNCTION__, __LINE__);
    }
  }

  /** Get config parameter as float.
   * \param id Identifier for config parameter
   */
  float getConfigFloat(const std::string &id) const
  {
    try
    {
      return config_.getValue<float>(id);
    }
    catch (std::exception &e)
    {
      // could not find the config parameter
      std::stringstream ss;
      ss << e.what() << std::endl;
      ss << "Operator '" << name_ << "': ";
      ss << "could not parse configuration parameter '" << id << "'!"
          << std::endl;
      throw IuException(ss.str(), __FILE__, __FUNCTION__, __LINE__);
    }
  }

  /** Get config parameter as int.
   * \param id Identifier for config parameter
   */
  int getConfigInt(const std::string &id) const
  {
    try
    {
      return config_.getValue<int>(id);
    }
    catch (std::exception &e)
    {
      // could not find the config parameter
      std::stringstream ss;
      ss << e.what() << std::endl;
      ss << "Operator '" << name_ << "': ";
      ss << "could not parse configuration parameter '" << id << "'!"
          << std::endl;
      throw IuException(ss.str(), __FILE__, __FUNCTION__, __LINE__);
    }
  }

  /** Get config parameter as string.
   * \param id Identifier for config parameter
   */
  std::string getConfigStr(const std::string &id) const
  {
    try
    {
      return config_.getStr(id);
    }
    catch (std::exception &e)
    {
      // could not find the config parameter
      std::stringstream ss;
      ss << e.what() << std::endl;
      ss << "Operator '" << name_ << "': ";
      ss << "could not parse configuration parameter '" << id << "'!"
          << std::endl;
      throw IuException(ss.str(), __FILE__, __FUNCTION__, __LINE__);
    }
  }

  /** Check if operator has a special config parameter.
   * \param id Identifier for config parameter
   * \return true if operator has config parameter.
   */
  bool hasConfig(const std::string &id) const
  {
    return config_.hasKey(id);
  }

private:
  /** Number of required operator constants. */
  const unsigned int num_required_constants_;

  /** Minimum number of required config parameters */
  int min_num_required_config_params_;

  /** Maximum number of required config parameters */
  int max_num_required_config_params_;

  /** Operator name */
  const std::string name_;

  /** Data container to store constants of iu::LinearDeviceMemory with arbitrary PixelType and Ndim. */
  std::vector<std::unique_ptr<IDataItem> > constants_;

  /** Make checks on the operator. Check if right number of constants is set. Make size check.
   * @param[in] in Source image of type InputType
   * @param[in] out Destination image of type OutputType
   */
  void operatorCheck(const InputType & in, const OutputType & out)
  {
    if (constants_.size() != num_required_constants_)
    {
      // too many constants
      std::stringstream ss;
      ss << "Operator '" << name_ << "': ";
      ss << "Number of constants (" << constants_.size()
          << ") does not match number of required constants ("
          << num_required_constants_ << ")";
      throw IuException(ss.str(), __FILE__, __FUNCTION__, __LINE__);
    }

    // check sizes of input, output and constants
    sizeCheck(in, out);
  }
};
